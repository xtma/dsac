import abc

import gtimer as gt
from rlkit.core import eval_util, logger
from rlkit.core.rl_algorithm import BaseRLAlgorithm, _get_epoch_timings
from rlkit.data_management.torch_replay_buffer import TorchReplayBuffer
from rlkit.samplers.data_collector import (VecMdpPathCollector,
                                           VecMdpStepCollector)


class VecOnlineRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: VecMdpStepCollector,
            evaluation_data_collector: VecMdpPathCollector,
            replay_buffer: TorchReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_paths_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_paths_per_epoch = num_eval_paths_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training

        assert self.num_trains_per_train_loop >= self.num_expl_steps_per_train_loop, \
            'Online training presumes num_trains_per_train_loop >= num_expl_steps_per_train_loop'

    def _train(self):
        self.training_mode(False)
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_steps(
                self.max_path_length,
                self.min_num_steps_before_training // self.expl_env.env_num,
                discard_incomplete_paths=False,
                random=True,  # whether random sample from action_space
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)
            gt.stamp('initial exploration', unique=True)

        num_trains_per_expl_step = self.num_trains_per_train_loop // self.num_expl_steps_per_train_loop
        num_trains_per_expl_step *= self.expl_env.env_num

        train_data = self.replay_buffer.next_batch(self.batch_size)
        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            for _ in range(self.num_train_loops_per_epoch):
                for _ in range(self.num_expl_steps_per_train_loop // self.expl_env.env_num):
                    new_expl_steps = self.expl_data_collector.collect_new_steps(
                        self.max_path_length,
                        1,  # num steps
                        discard_incomplete_paths=False,
                    )
                    gt.stamp('exploration sampling', unique=False)
                    self.replay_buffer.add_paths(new_expl_steps)
                    gt.stamp('data storing', unique=False)

                    self.training_mode(True)
                    for _ in range(num_trains_per_expl_step):
                        self.trainer.train(train_data)
                        gt.stamp('training', unique=False)
                        train_data = self.replay_buffer.next_batch(self.batch_size)
                        gt.stamp('data sampling', unique=False)
                    self.training_mode(False)

            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_paths_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling')

            self._end_epoch(epoch)

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        return snapshot

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        logger.record_dict(
            self.replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(),
            prefix='exploration/'
        )
        """
        Evaluation
        """
        logger.record_dict(
            self.eval_data_collector.get_diagnostics(),
            prefix='evaluation/',
        )
        eval_paths = self.eval_data_collector.get_epoch_paths()
        logger.record_dict(
            eval_util.get_generic_path_information(eval_paths),
            prefix="evaluation/",
        )

        """
        Misc
        """
        gt.stamp('logging')
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
