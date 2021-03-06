This repo contains a basic implementation of the [A3C algorithm](https://arxiv.org/abs/1602.01783), adapted for real-time environments.

# Dependencies

* Python 2.7 or 3.5
* [Golang](https://golang.org/doc/install)
* [six](https://pypi.python.org/pypi/six) (for py2/3 compatibility)
* [TensorFlow](https://www.tensorflow.org/) 1.5+
* [tmux](https://tmux.github.io/) (the start script opens up a tmux session with multiple windows)
* [htop](https://hisham.hm/htop/) (shown in one of the tmux windows)
* [gym](https://pypi.python.org/pypi/gym)
* libjpeg-turbo (`brew install libjpeg-turbo`)
* [universe](https://pypi.python.org/pypi/universe)
* [opencv-python](https://pypi.python.org/pypi/opencv-python)
* [numpy](https://pypi.python.org/pypi/numpy)
* [scipy](https://pypi.python.org/pypi/scipy)

`python train.py --num-workers 4 --env-id flashgames.NeonRace-v0 --log-dir ~/neonrace`

The command above will train an agent
It will see 4 workers that will be learning in parallel (`--num-workers` flag) and will output intermediate results into given directory.

The code will launch the following processes:
* worker-0 to worker 3 - four processes that run policy gradient
* ps - the parameter server, which synchronizes the parameters among the different workers
* tb - a tensorboard process for convenient display of the statistics of learning

Once you start the training process, it will create a tmux session with a window for each of these processes. You can connect to them by typing `tmux a` in the console.
Once in the tmux session, you can see all your windows with `ctrl-b w`.
To switch to window number 0, type: `ctrl-b 0`. Look up tmux documentation for more commands.

To access TensorBoard to see various monitoring metrics of the agent, open [http://localhost:12345/](http://localhost:12345/) in a browser.

Add '--visualise' toggle if you want to visualise the worker using env.render() as follows:

`python train.py --num-workers 4 --env-id flashgames.NeonRace-v0 --log-dir ~/neonrace --visualise`

