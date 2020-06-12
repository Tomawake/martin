import numpy as np

from lib.RLTrader import RLTrader
from lib.cli.RLTraderCLI import RLTraderCLI
from lib.util.logger import init_logger
from lib.env.reward import BaseRewardStrategy, IncrementalProfit, WeightedUnrealizedProfit


#np.warnings.filterwarnings('ignore')

trader_cli = RLTraderCLI()
args = trader_cli.get_args()

rewards = {"incremental-profit": IncrementalProfit, "weighted-unrealized-profit": WeightedUnrealizedProfit}
reward_strategy = rewards[args.reward_strat]

if __name__ == '__main__':
    logger = init_logger(__name__, show_debug=args.debug)
    trader = RLTrader(**vars(args), logger=logger, reward_strategy=reward_strategy)

    trader.train(n_epochs=10,
                 save_every=1,
                 test_trained_model="store_false",
                 render_test_env="store_true",
                 render_report="store_false",
                 save_report="store_true")