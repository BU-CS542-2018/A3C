#--------------------------------------------------------------------------------------------------------------------------------
# CS 542 Machine Learning Project, Winter 2018, Boston University
# Modified for the purpose of project
# Original code by OpenAI
# Description: Implementation of the workers, each of them are executed through tensorflow parallel work within 
# independent tmux session.
#--------------------------------------------------------------------------------------------------------------------------------
#!/usr/bin/env python
import cv2
import go_vncdriver
import tensorflow as tf
import argparse
import logging
import sys, signal
import time
import os
from a3c import A3C
from envs import create_env
import distutils.version
use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
#--------------------------------------------------------------------------------------------------------------------------------
# Disables write_meta_graph argument, which freezes entire process and is mostly useless.
class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)
#--------------------------------------------------------------------------------------------------------------------------------
# This is where the worker trains the model
def run(args, server):
    #environment and trainer
    env = create_env(args.env_id, client_id=str(args.task), remotes=args.remotes)
    trainer = A3C(env, args.task, args.visualise)

    # Variable names that start with "local" are not saved in checkpoints.
    # global_variables are shared between distributed machines
    if use_tf12_api:
        variables_to_save = [v for v in tf.global_variables() if not v.name.startswith("local")]
        init_op = tf.variables_initializer(variables_to_save)
        init_all_op = tf.global_variables_initializer()
    # this will not be run since we are using a latest version of tensorflow
    else:
        variables_to_save = [v for v in tf.all_variables() if not v.name.startswith("local")]
        init_op = tf.initialize_variables(variables_to_save)
        init_all_op = tf.initialize_all_variables()
    #saver for saving the parameters
    saver = FastSaver(variables_to_save)
    
    #get trainable 
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    logger.info('Trainable vars:')
    for v in var_list:
        logger.info('  %s %s', v.name, v.get_shape())
    #--------------------------------------------------------------------------------------------------------------------------------
    #
    def init_fn(ses):
        logger.info("Initializing all parameters.")
        ses.run(init_all_op)
    #--------------------------------------------------------------------------------------------------------------------------------
    config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(args.task)])
    logdir = os.path.join(args.log_dir, 'train')

    if use_tf12_api:
        summary_writer = tf.summary.FileWriter(logdir + "_%d" % args.task)
    else:
        summary_writer = tf.train.SummaryWriter(logdir + "_%d" % args.task)

    logger.info("Events directory: %s_%s", logdir, args.task)
    # 
    # https://www.tensorflow.org/api_docs/python/tf/train/Supervisor
    sv = tf.train.Supervisor(is_chief=(args.task == 0),
                             logdir=logdir,
                             saver=saver,
                             summary_op=None,
                             init_op=init_op,
                             init_fn=init_fn,
                             summary_writer=summary_writer,
                             ready_op=tf.report_uninitialized_variables(variables_to_save),
                             global_step=trainer.global_step,
                             save_model_secs=30,
                             save_summaries_secs=30)

    #maximum amount of steps for the 
    num_global_steps = 100000000

    logger.info(
        "Starting session. If this hangs, we're mostly likely waiting to connect to the parameter server. " +
        "One common cause is that the parameter server DNS name isn't resolving yet, or is misspecified.")
    with sv.managed_session(server.target, config=config) as sess, sess.as_default():
        sess.run(trainer.sync)
        trainer.start(sess, summary_writer)
        global_step = sess.run(trainer.global_step)
        logger.info("Starting training at step=%d", global_step)
        
        # This is the training loop, 
        while not sv.should_stop() and (not num_global_steps or global_step < num_global_steps):
            trainer.process(sess)
            global_step = sess.run(trainer.global_step)

    # Ask for all the services to stop.
    sv.stop()
    logger.info('reached %s steps. worker stopped.', global_step)
#--------------------------------------------------------------------------------------------------------------------------------
def cluster_spec(num_workers, num_ps):
    """
More tensorflow setup for data parallelism
"""
    cluster = {} # this is a dictionary
    port = 12222  #first port address
    
    #generate all parameter space IP port spec
    all_ps = []
    host = '127.0.0.1'           #local IP address
    for _ in range(num_ps):
        all_ps.append('{}:{}'.format(host, port))
        port += 1
    cluster['ps'] = all_ps
    
    #generate all worker IP port spec
    all_workers = []
    for _ in range(num_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1
    cluster['worker'] = all_workers
    # in other words, the PS will start from 12222 to 12222 + num_ps - 1 and worker will start from 12222 + num_ps to 
    # 12222 + num_ps + num_workers - 1
    return cluster            
#--------------------------------------------------------------------------------------------------------------------------------
def main(_):
    """
Setting up Tensorflow for data parallel work
"""
    # parse the cmd passed from train.py
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('--task', default=0, type=int, help='Task index')
    parser.add_argument('--job-name', default="worker", help='worker or ps')
    parser.add_argument('--num-workers', default=1, type=int, help='Number of workers')
    parser.add_argument('--log-dir', default="/tmp/neonrace", help='Log directory path')
    parser.add_argument('--env-id', default="flashgames.NeonRace-v0", help='Environment id')
    parser.add_argument('-r', '--remotes', default=None,
                        help='References to environments to create (e.g. -r 20), '
                             'or the address of pre-existing VNC servers and '
                             'rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901)')

    # Add visualisation argument
    parser.add_argument('--visualise', action='store_true',
                        help="Visualise the gym environment by running env.render() between each timestep")

    args = parser.parse_args()
    
    spec = cluster_spec(args.num_workers, 1)   # n workers, 1 parameter updater
    # return the cluster def object with the specification
    # https://www.tensorflow.org/api_docs/python/tf/train/ClusterSpec
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()
    #--------------------------------------------------------------------------------------------------------------------------------
    #inline function
    def shutdown(signal, frame):
        logger.warn('Received signal %s: exiting', signal)
        sys.exit(128+signal)
    #--------------------------------------------------------------------------------------------------------------------------------
    # exception handling
    signal.signal(signal.SIGHUP, shutdown)  #hang up signal
    signal.signal(signal.SIGINT, shutdown)  #keyboard interrupt exception
    signal.signal(signal.SIGTERM, shutdown) #terminat signal
    
    # execute 
    if args.job_name == "worker":
        #https://www.tensorflow.org/api_docs/python/tf/train/Server
        # configuration protocol
        server = tf.train.Server(cluster, job_name="worker", task_index=args.task,
                                 config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2))
        run(args, server)  #train workers
    else:
        server = tf.train.Server(cluster, job_name="ps", task_index=args.task,
                                 config=tf.ConfigProto(device_filters=["/job:ps"]))
        #busy waiting
        while True:
            time.sleep(1000)
#--------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    tf.app.run()
