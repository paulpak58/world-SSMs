# Adopted from https://github.com/danijar/dreamerv3/blob/main/dreamerv3/train.py
# Revised for TurtleSim/SpotSim

import os
import sys
import warnings
import argparse 
import dreamerv3
from dreamerv3 import embodied
from dreamerv3 import agent as agt
from dreamerv3.embodied import wrappers
from dreamerv3.embodied.envs import from_gym
from functools import partial
from importlib import import_module
import pathlib


directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')


#################################
# Wrap the observation and action
# and clip if necessary
#################################
def wrap_env(env, config):
	args = config.wrapper
	for name, space in env.act_space.items():
		if name=='reset':
			continue
		elif space.discrete:
			env = wrappers.OneHotAction(env, name)
		elif args.discretize:
			env = wrappers.DiscretizeAction(env, name, args.discretize)
		else:
			env = wrappers.NormalizeAction(env, name)
	env = wrappers.ExpandScalars(env)
	if args.length:
		env = wrappers.TimeLimit(env, args.length, args.reset)
	if args.checks:
		env = wrappers.CheckSpaces(env)
	for name, space in env.act_space.items():
		if not space.discrete:
			env = wrappers.ClipAction(env, name)
	return env


########################################
# Logger for storing history and results
########################################
def make_logger(parsed, logdir, step ,config):
	multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
	logger = embodied.Logger(step, [
		embodied.logger.TerminalOutput(config.filter),
		embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
		embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score'),
		embodied.logger.TensorBoardOutput(logdir),
		# embodied.logger.WandBOutput(logdir.name, config),
		# embodied.logger.MLFlowOutput(logdir.name),
	], multiplier)
	return logger


########################################
# Replay buffer for storing trajectories
########################################
def make_replay(config, directory=None, is_eval=False, rate_limit=False, **kwargs):
	assert config.replay=='uniform' or not rate_limit
	length = config.batch_length
	size = config.replay_size//10 if is_eval else config.replay_size
	if config.replay=='uniform' or is_eval:
		kw = {'online': config.replay_online}
		if rate_limit and config.run.train_ratio > 0:
			kw['samples_per_insert'] = config.run.train_ratio/config.batch_length
			kw['tolerance'] = 10*config.batch_size
			kw['min_size'] = config.batch_size
		replay = embodied.replay.Uniform(length, size, directory, **kw)
	elif config.replay=='reverb':
		replay = embodied.replay.Reverb(length, size, directory)
	elif config.replay=='chunks':
		replay = embodied.replay.NaiveChunks(length, size, directory)
	else:
		raise NotImplementedError(config.replay)
	return replay



#####################################
# Default batched environment wrapper
#####################################
def make_envs(config, **overrides):
	suite, task = config.task.split('_', 1)
	env_suite = []
	for index in range(config.envs.amount):
		e = lambda: make_env(config, **overrides)
		if config.envs.parallel != 'none':
			e = partial(embodied.Parallel, e, config.envs.parallel)
		if config.envs.restart:
			e = partial(wrappers.RestartOnException, e)
		env_suite.append(e)
	envs = [e() for e in env_suite]
	return embodied.BatchEnv(envs, parallel=(config.envs.parallel != 'none'))


###################################
# Create custom environments here
###################################
def make_env(config, **overrides):
	# use `embodied.envs.from_gym.FromGym` for custom interfaces
	suite, task = config.task.split('_', 1)
	e = {
	  'dummy': 'embodied.envs.dummy:Dummy',
	  'gym': 'embodied.envs.from_gym:FromGym',
	  'dm': 'embodied.envs.from_dmenv:FromDM',
	  'crafter': 'embodied.envs.crafter:Crafter',
	  'dmc': 'embodied.envs.dmc:DMC',
	  'atari': 'embodied.envs.atari:Atari',
	  'dmlab': 'embodied.envs.dmlab:DMLab',
	  'minecraft': 'embodied.envs.minecraft:Minecraft',
	  'loconav': 'embodied.envs.loconav:LocoNav',
	  'pinpad': 'embodied.envs.pinpad:PinPad',
	}[suite]
	if isinstance(e, str):
		module, cls = e.split(':')
		module = import_module(module)
		e = getattr(module, cls)
	kwargs = config.env.get(suite, {})
	kwargs.update(overrides)
	env = e(task, **kwargs)
	return wrap_env(env, config)






def main(argv=None):

	parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
	config = embodied.Config(agt.Agent.configs['defaults'])
	for name in parsed.configs:
		config = config.update(agt.Agent.configs[name])

	# First, update with environment and then model size
	#config = config.update(agt.Agent.configs['atari'])
	#config = config.update(agt.Agent.configs['medium'])
	config = embodied.Flags(config).parse(other)
	args = embodied.Config(**config.run, logdir=config.logdir, batch_steps=config.batch_size*config.batch_length)
	print(config)

	# Set up logging
	logdir = embodied.Path(args.logdir)
	logdir.mkdirs()
	config.save(logdir/'config.yaml')
	step = embodied.Counter()
	logger = make_logger(parsed, logdir, step, config)


	cleanup = []
	try:
		env = make_envs(config)
		if args.script=='train':
			replay = make_replay(config, logdir/'replay')
			cleanup.append(env)
			agent = agt.Agent(env.obs_space, env.act_space, step, config)
			embodied.run.train(agent, env, replay, logger, args)
		elif args.script=='train_save':
			replay = make_replay(config, logdir/'replay')
			cleanup.append(env)
			agent = agt.Agent(env.obs_space, env.act_space, step, config)
			embodied.run.train_save(agent, env, replay, logger, args)
		elif args.script=='train_eval':
			replay = make_replay(config, logdir/'replay')
			eval_replay = make_replay(config, logdir/'eval_replay', is_eval=True)
			eval_env = make_envs(config)
			cleanup += [env, eval_env]
			agent = agt.Agent(env.obs_space, env.act_space, step, config)
			embodied.run.train_eval(agent, env, eval_env, replay, eval_replay, logger, args)
		elif args.script=='train_holdout':
			replay = make_replay(config, logdir/'replay')
			if config.eval_dir:
				assert not config.train.eval_fill
				eval_replay = make_replay(config, config.eval_dir, is_eval=True)
			else:
				assert 0 < args.eval_fill <= config.replay_size//10, args.eval_fill
				eval_replay = make_replay(config, logdir/'eval_replay', is_eval=True)
			cleanup.append(env)
			agent = agt.Agent(env.obs_space, env.act_space, step, config)
			embodied.run.train_holdout(agent, env, replay, eval_replay, logger, args)
		elif args.script=='eval_only':
			cleanup.append(env)
			agent = agt.Agent(env.obs_space, env.act_space, step, config)
			embodied.run.eval_only(agent, env, logger, args)
		elif args.script=='parallel':
			assert config.run.actor_batch<=config.envs.amount, (config.run.actor_batch, config.envs.amount)
			step = embodied.Counter()
			agent = agt.Agent(env.obs_space, env.act_space, step, config)
			env.close()
			replay = make_replay(config, logdir/'replay', rate_limit=True)
			embodied.run.parallel(agent, replay, logger, partial(make_envs, config), num_envs=config.envs.amount, args=args)
		else:
			raise NotImplementedError(args.script)
	finally:
		for obj in cleanup:
			obj.close()





if __name__=='__main__':
	# import argparse
	# parser = argparse.ArgumentParser()
	# parser.add_argument('--world', default='atari', choices=['atari', 'dmlab', 'minecraft', 'loconav', 'pinpad'], help='The environment')
	# parser.add_argument('--model_size', default='small', choices=['small', 'medium', 'large'], help='The model size')
	# args, config_args = parser.parse_known_args()
	main()
