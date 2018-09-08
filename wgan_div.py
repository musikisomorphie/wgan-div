import os
import sys

sys.path.append(os.getcwd())

import time
import functools

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.data_loader
import tflib.ops.layernorm
import tflib.plot

import fid
import re

DATA_DIR = 'path/to/file'
DATASET = "celeba"  # celeba, cifar10, svhn, lsun
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_64x64.py!')

# Download the Inception model from here
# http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
# And set the path to the extracted model here:
INCEPTION_DIR = "inception-2015-12-05"

# Path to the real world statistics file.
STAT_FILE = "stats/fid_stats_celeba.npz"
LOAD_CHECKPOINT = False

LOG_DIR = "logs"  # Directory for Tensorboard events, checkpoints and samples
N_GPUS = 1  # Number of GPUs

# Model hyperparamters
MODE = 'wgan-div'  # dcgan, wgan, wgan-div, lsgan
DIM = 64  # Model dimensionality
OUTPUT_DIM = DIM * DIM * 3  # Number of pixels in each image
BN_D = True
BN_G = True
NON_LIN = tf.nn.relu

ITER_START = 0
ITERS = 200000  # How many iterations to train for
CRITIC_ITERS = 4
BATCH_SIZE = 64  # Batch size. Must be a multiple of N_GPUS
LAMBDA = 10  # Gradient penalty lambda hyperparameter
LR = 2e-4  # Initial learning rate

# Print steps
OUTPUT_STEP = 400  # Print output every OUTPUT_STEP
SAVE_SAMPLES_STEP = 400  # Generate and save samples every SAVE_SAMPLES_STEP
CHECKPOINT_STEP = 5000  # FID_STEP

# FID evaluation.
FID_STEP = 1000
FID_EVAL_SIZE = 50000  # Number of samples for evaluation
FID_SAMPLE_BATCH_SIZE = 1000  # Batch size of generating samples, lower to save GPU memory
FID_BATCH_SIZE = 200  # Batch size for final FID calculation i.e. inception propagation etc.

# Process Checkpoint and Save paths
if not LOAD_CHECKPOINT:
    timestamp = time.strftime("%m%d_%H%M%S")
    DIR = "%s_%6f_wgan-div" % (timestamp, LR)
else:
    DIR = "%s_%6f" % ('1224_174334', LR)

LOG_DIR = os.path.join(LOG_DIR, DIR)
SAMPLES_DIR = os.path.join(LOG_DIR, "samples")
CHECKPOINT_DIR = os.path.join(LOG_DIR, "checkpoints")
TBOARD_DIR = os.path.join(LOG_DIR, "logs")

# Create directories if necessary
if not os.path.exists(SAMPLES_DIR):
    print("*** create sample dir %s" % SAMPLES_DIR)
    os.makedirs(SAMPLES_DIR)
if not os.path.exists(CHECKPOINT_DIR):
    print("*** create checkpoint dir %s" % CHECKPOINT_DIR)
    os.makedirs(CHECKPOINT_DIR)
if not os.path.exists(TBOARD_DIR):
    print("*** create tboard dir %s" % TBOARD_DIR)
    os.makedirs(TBOARD_DIR)


# Load checkpoint
# from https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
def load_checkpoint(session, saver, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    print(checkpoint_dir)
    i = 0

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(session, os.path.join(checkpoint_dir, ckpt_name))

        latest_cp = tf.train.latest_checkpoint(checkpoint_dir)

        i = int(re.findall('\d+', latest_cp)[-1]) + 1

        print(" [*] Success to read {}".format(ckpt_name))
        return True, i
    else:
        print(" [*] Failed to find a checkpoint")
        return False, i


def GeneratorAndDiscriminator():
    """
    Choose which generator and discriminator architecture to use by
    uncommenting one of these lines.
    """
    return GoodGenerator, GoodDiscriminator
    # return DCGANGenerator, DCGANDiscriminator
    raise Exception('You must choose an architecture!')


DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


def ReLULayer(name, n_in, n_out, inputs):
    output = tflib.ops.linear.Linear(name + '.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)


def LeakyReLULayer(name, n_in, n_out, inputs):
    output = tflib.ops.linear.Linear(name + '.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)


def Normalize(name, axes, inputs):
    # return inputs
    if ('Discriminator' in name) and (MODE == 'wgan-div'):
        if axes != [0, 2, 3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return lib.ops.layernorm.Layernorm(name, [1, 2, 3], inputs)
    else:
        return lib.ops.batchnorm.Batchnorm(name, axes, inputs, fused=True)


def pixcnn_gated_nonlinearity(a, b):
    return tf.sigmoid(a) * tf.tanh(b)


def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4 * kwargs['output_dim']
    output = tflib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0, 3, 1, 2])
    return output


def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = tflib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n(
        [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    return output


def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n(
        [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    output = tflib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output


def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0, 3, 1, 2])
    output = tflib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output


def BottleneckResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample == 'down':
        conv_shortcut = functools.partial(tflib.ops.conv2d.Conv2D, stride=2)
        conv_1 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim // 2)
        conv_1b = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim // 2, output_dim=output_dim // 2,
                                    stride=2)
        conv_2 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=output_dim // 2, output_dim=output_dim)
    elif resample == 'up':
        conv_shortcut = SubpixelConv2D
        conv_1 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim // 2)
        conv_1b = functools.partial(tflib.ops.deconv2d.Deconv2D, input_dim=input_dim // 2, output_dim=output_dim // 2)
        conv_2 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=output_dim // 2, output_dim=output_dim)
    elif resample == None:
        conv_shortcut = tflib.ops.conv2d.Conv2D
        conv_1 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim // 2)
        conv_1b = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim // 2, output_dim=output_dim // 2)
        conv_2 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim // 2, output_dim=output_dim)

    else:
        raise Exception('invalid resample value')

    if output_dim == input_dim and resample == None:
        shortcut = inputs  # Identity skip-connection
    else:
        shortcut = conv_shortcut(name + '.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = tf.nn.relu(output)
    output = conv_1(name + '.Conv1', filter_size=1, inputs=output, he_init=he_init)
    output = tf.nn.relu(output)
    output = conv_1b(name + '.Conv1B', filter_size=filter_size, inputs=output, he_init=he_init)
    output = tf.nn.relu(output)
    output = conv_2(name + '.Conv2', filter_size=1, inputs=output, he_init=he_init, biases=False)
    output = Normalize(name + '.BN', [0, 2, 3], output)

    return shortcut + (0.3 * output)


def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True, bn=False):
    """
    resample: None, 'down', or 'up'
    """
    if resample == 'down':
        conv_shortcut = MeanPoolConv
        conv_1 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2 = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
    elif resample == 'up':
        conv_shortcut = UpsampleConv
        conv_1 = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_2 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample == None:
        conv_shortcut = tflib.ops.conv2d.Conv2D
        conv_1 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim == input_dim and resample == None:
        shortcut = inputs  # Identity skip-connection
    else:
        shortcut = conv_shortcut(name + '.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    if bn:
        output = Normalize(name + '.BN1', [0, 2, 3], output)
    output = tf.nn.relu(output)
    output = conv_1(name + '.Conv1', filter_size=filter_size, inputs=output, he_init=he_init, biases=False)
    if bn:
        output = Normalize(name + '.BN2', [0, 2, 3], output)
    output = tf.nn.relu(output)
    output = conv_2(name + '.Conv2', filter_size=filter_size, inputs=output, he_init=he_init)

    return shortcut + output


# ! Generators
def GoodGenerator(n_samples, noise=None, dim=DIM, nonlinearity=tf.nn.relu, bn=BN_G):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    ## supports 32x32 images
    fact = DIM // 16

    output = lib.ops.linear.Linear('Generator.Input', 128, fact * fact * 8 * dim, noise)
    output = tf.reshape(output, [-1, 8 * dim, fact, fact])
    output = ResidualBlock('Generator.Res1', 8 * dim, 8 * dim, 3, output, resample='up', bn=bn)
    output = ResidualBlock('Generator.Res2', 8 * dim, 4 * dim, 3, output, resample='up', bn=bn)
    output = ResidualBlock('Generator.Res3', 4 * dim, 2 * dim, 3, output, resample='up', bn=bn)
    output = ResidualBlock('Generator.Res4', 2 * dim, 1 * dim, 3, output, resample='up', bn=bn)
    if bn:
        output = Normalize('Generator.OutputN', [0, 2, 3], output)
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', 1 * dim, 3, 3, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])


# ! Discriminators
def GoodDiscriminator(inputs, dim=DIM, bn=BN_D):
    output = tf.reshape(inputs, [-1, 3, DIM, DIM])
    output = lib.ops.conv2d.Conv2D('Discriminator.Input', 3, dim, 3, output, he_init=False)

    output = ResidualBlock('Discriminator.Res1', dim, 2 * dim, 3, output, resample='down', bn=bn)
    output = ResidualBlock('Discriminator.Res2', 2 * dim, 4 * dim, 3, output, resample='down', bn=bn)
    output = ResidualBlock('Discriminator.Res3', 4 * dim, 8 * dim, 3, output, resample='down', bn=bn)
    output = ResidualBlock('Discriminator.Res4', 8 * dim, 8 * dim, 3, output, resample='down', bn=bn)

    output = tf.reshape(output, [-1, 4 * 4 * 8 * dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 4 * 4 * 8 * dim, 1, output)

    return tf.reshape(output, [-1])


def DCGANGenerator(n_samples, noise=None, dim=DIM, bn=BN_G, nonlinearity=NON_LIN):
    lib.ops.conv2d.set_weights_stdev(0.02)
    lib.ops.deconv2d.set_weights_stdev(0.02)
    lib.ops.linear.set_weights_stdev(0.02)

    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4 * 4 * 8 * dim, noise)
    output = tf.reshape(output, [-1, 8 * dim, 4, 4])
    if bn:
        output = Normalize('Generator.BN1', [0, 2, 3], output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 8 * dim, 4 * dim, 5, output)
    if bn:
        output = Normalize('Generator.BN2', [0, 2, 3], output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 4 * dim, 2 * dim, 5, output)
    if bn:
        output = Normalize('Generator.BN3', [0, 2, 3], output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.4', 2 * dim, dim, 5, output)
    if bn:
        output = Normalize('Generator.BN4', [0, 2, 3], output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
    output = tf.tanh(output)

    lib.ops.conv2d.unset_weights_stdev()
    lib.ops.deconv2d.unset_weights_stdev()
    lib.ops.linear.unset_weights_stdev()

    return tf.reshape(output, [-1, OUTPUT_DIM])


def DCGANDiscriminator(inputs, dim=DIM, bn=BN_D, nonlinearity=NON_LIN):
    output = tf.reshape(inputs, [-1, 3, DIM, DIM])

    lib.ops.conv2d.set_weights_stdev(0.02)
    lib.ops.deconv2d.set_weights_stdev(0.02)
    lib.ops.linear.set_weights_stdev(0.02)

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, dim, 5, output, stride=2)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', dim, 2 * dim, 5, output, stride=2)
    if bn:
        output = Normalize('Discriminator.BN2', [0, 2, 3], output)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2 * dim, 4 * dim, 5, output, stride=2)
    if bn:
        output = Normalize('Discriminator.BN3', [0, 2, 3], output)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.4', 4 * dim, 8 * dim, 5, output, stride=2)
    if bn:
        output = Normalize('Discriminator.BN4', [0, 2, 3], output)
    output = nonlinearity(output)

    output = tf.reshape(output, [-1, 4 * 4 * 8 * dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 4 * 4 * 8 * dim, 1, output)

    lib.ops.conv2d.unset_weights_stdev()
    lib.ops.deconv2d.unset_weights_stdev()
    lib.ops.linear.unset_weights_stdev()

    return tf.reshape(output, [-1])


Generator, Discriminator = GeneratorAndDiscriminator()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    all_real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, DIM, DIM])
    if tf.__version__.startswith('1.'):
        split_real_data_conv = tf.split(all_real_data_conv, len(DEVICES))
    else:
        split_real_data_conv = tf.split(0, len(DEVICES), all_real_data_conv)
    gen_costs, disc_costs, recon_costs = [], [], []

    for device_index, (device, real_data_conv) in enumerate(zip(DEVICES, split_real_data_conv)):
        with tf.device(device):
            real_data = tf.reshape(2 * ((tf.cast(real_data_conv, tf.float32) / 255.) - .5),
                                   [BATCH_SIZE / len(DEVICES), OUTPUT_DIM])
            fake_data = Generator(BATCH_SIZE / len(DEVICES))

            disc_real = Discriminator(real_data)
            disc_fake = Discriminator(fake_data)

            gen_cost = tf.reduce_mean(disc_fake)
            disc_cost = -tf.reduce_mean(disc_fake) + tf.reduce_mean(disc_real)

            alpha = tf.random_uniform(
                shape=[BATCH_SIZE / len(DEVICES), 1],
                minval=0.,
                maxval=1.
            )
            differences = fake_data - real_data
            interpolates = real_data + (alpha * differences)
            gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
            slopes = tf.pow(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]), 3)
            gradient_penalty = tf.reduce_mean(slopes)
            disc_cost += LAMBDA * gradient_penalty

            gen_costs.append(gen_cost)
            disc_costs.append(disc_cost)

    gen_cost = tf.add_n(gen_costs) / len(DEVICES)
    disc_cost = tf.add_n(disc_costs) / len(DEVICES)

    gen_train_op = tf.train.AdamOptimizer(learning_rate=LR, beta1=0., beta2=0.9).minimize(gen_cost,
                                                                                          var_list=lib.params_with_name(
                                                                                              'Generator'),
                                                                                          colocate_gradients_with_ops=True)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=LR, beta1=0., beta2=0.9).minimize(disc_cost,
                                                                                           var_list=lib.params_with_name(
                                                                                               'Discriminator.'),
                                                                                           colocate_gradients_with_ops=True)
    tf.summary.scalar("gen_cost", gen_cost)
    tf.summary.scalar("disc_cost", disc_cost)

    summary_op = tf.summary.merge_all()

    # For generating samples
    fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
    all_fixed_noise_samples = []
    for device_index, device in enumerate(DEVICES):
        n_samples = BATCH_SIZE // len(DEVICES)
        all_fixed_noise_samples.append(Generator(n_samples,
                                                 noise=fixed_noise[
                                                       device_index * n_samples:(device_index + 1) * n_samples]))
    if tf.__version__.startswith('1.'):
        all_fixed_noise_samples = tf.concat(all_fixed_noise_samples, axis=0)
    else:
        all_fixed_noise_samples = tf.concat(0, all_fixed_noise_samples)


    def generate_image(iteration):
        samples = session.run(all_fixed_noise_samples)
        samples = ((samples + 1.) * (255.99 // 2)).astype('int32')
        tflib.save_images.save_images(samples.reshape((BATCH_SIZE, 3, DIM, DIM)),
                                      '%s/samples_%d.png' % (SAMPLES_DIR, iteration))


    fid_tfvar = tf.Variable(0.0, trainable=False)
    fid_sum = tf.summary.scalar("FID", fid_tfvar)

    writer = tf.summary.FileWriter(TBOARD_DIR, session.graph)

    # Dataset iterator
    train_gen, dev_gen = tflib.data_loader.load(BATCH_SIZE, DATA_DIR, DATASET)


    def inf_train_gen():
        while True:
            for (images,) in train_gen():
                yield images


    # Save a batch of ground-truth samples
    _x = inf_train_gen().next()
    _x_r = session.run(real_data, feed_dict={real_data_conv: _x[:BATCH_SIZE // N_GPUS]})
    _x_r = ((_x_r + 1.) * (255.99 // 2)).astype('int32')
    tflib.save_images.save_images(_x_r.reshape((BATCH_SIZE // N_GPUS, 3, DIM, DIM)),
                                  '%s/samples_groundtruth.png' % SAMPLES_DIR)

    session.run(tf.global_variables_initializer())

    # Checkpoint saver
    ckpt_saver = tf.train.Saver(max_to_keep=int(ITERS / CHECKPOINT_STEP))

    if LOAD_CHECKPOINT:
        is_check, ITER_START = load_checkpoint(session, ckpt_saver, CHECKPOINT_DIR)
        if is_check:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

    gen = inf_train_gen()

    # load model
    # print("load inception model..", end=" ", flush=True)
    print("load inception model..")

    fid.create_inception_graph(os.path.join(INCEPTION_DIR, "classify_image_graph_def.pb"))
    print("ok")

    # print("load train stats.. ", end="", flush=True)
    print("load train stats.. ")

    # load precalculated training set statistics
    f = np.load(STAT_FILE)
    mu_real, sigma_real = f['mu'][:], f['sigma'][:]
    f.close()
    print("ok")

    # Train loop

    for it in range(ITERS):

        iteration = it + ITER_START

        start_time = time.time()

        # Train critic
        if iteration > 0:
            _gen_cost, _ = session.run([gen_cost, gen_train_op])

        for i in xrange(CRITIC_ITERS):
            _data = gen.next()
            _disc_cost, _, _summary_op = session.run([disc_cost, disc_train_op, summary_op],
                                                     feed_dict={all_real_data_conv: _data})

        if iteration % SAVE_SAMPLES_STEP == SAVE_SAMPLES_STEP - 1:
            generate_image(iteration)
            print("Time: %g/itr, Itr: %d, generator loss: %g , discriminator_loss: %g" % (
                time.time() - start_time, iteration, _gen_cost, _disc_cost))
            writer.add_summary(_summary_op, iteration)

        if iteration % FID_STEP == FID_STEP - 1:
            # FID
            samples = np.zeros((FID_EVAL_SIZE, OUTPUT_DIM), dtype=np.uint8)

            n_fid_batches = FID_EVAL_SIZE // FID_SAMPLE_BATCH_SIZE

            for i in range(n_fid_batches):
                frm = i * FID_SAMPLE_BATCH_SIZE
                to = frm + FID_SAMPLE_BATCH_SIZE
                tmp = session.run(Generator(FID_SAMPLE_BATCH_SIZE))
                samples[frm:to] = ((tmp + 1.0) * 127.5).astype('uint8')

            # Cast, reshape and transpose (BCHW -> BHWC)
            samples = samples.reshape(FID_EVAL_SIZE, 3, DIM, DIM)
            samples = samples.transpose(0, 2, 3, 1)

            print("ok")

            mu_gen, sigma_gen = fid.calculate_activation_statistics(samples,
                                                                    session,
                                                                    batch_size=FID_BATCH_SIZE,
                                                                    verbose=True)

            try:
                FID = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
            except Exception as e:
                print(e)
                FID = 500

            print("calculate FID: %f " % (FID))
            session.run(tf.assign(fid_tfvar, FID))
            summary_str = session.run(fid_sum)
            writer.add_summary(summary_str, iteration)

        # Save checkpoint
        if iteration % CHECKPOINT_STEP == CHECKPOINT_STEP - 1:
            if iteration == CHECKPOINT_STEP - 1:
                ckpt_saver.save(session,
                                os.path.join(CHECKPOINT_DIR, "wgan-div.model"),
                                iteration, write_meta_graph=True)
            else:
                ckpt_saver.save(session,
                                os.path.join(CHECKPOINT_DIR, "wgan-div.model"),
                                iteration, write_meta_graph=False)
