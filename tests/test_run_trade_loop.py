import os
import sys
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import trading_bot.run_trade_loop as rl


class DummySentiment:
    def get_sentiment_score(self, text):
        return 0.0


class DummyModel:
    def __init__(self, input_dim):
        pass

    def load_state_dict(self, state):
        pass

    def eval(self):
        pass

    def __call__(self, x):
        class Pred:
            def item(self_inner):
                return 0.6

        return Pred()


class DummyTorch:
    float32 = None

    def tensor(self, data, dtype=None):
        return data

    class no_grad:
        def __enter__(self_inner):
            pass

        def __exit__(self_inner, exc_type, exc, tb):
            pass



def test_run_trade_loop_submits_order_on_signal():
    with patch.object(rl, "SentimentAnalyzer", return_value=DummySentiment()):
        with patch.object(rl, "LSTMPricePredictor", DummyModel):
            with patch.object(rl, "torch", DummyTorch()):
                with patch.object(rl.order_submitter, "submit_bracket_order") as mock_submit:
                    with patch.object(rl.position_manager, "get_open_position", return_value=None):
                        with patch.object(rl.volatility_manager, "compute_stop_target", return_value=(0.01, 0.02)):
                            with patch.object(rl.position_sizer, "calculate_position_size", return_value=5):
                                with patch.object(rl, "_fetch_equity", return_value=10000):
                                    with patch.object(rl.time, "sleep", return_value=None):
                                        rl.run_trade_loop(
                                            symbol="AAPL",
                                            seq_len=1,
                                            price_fetcher=lambda s: 100.0,
                                            news_fetcher=lambda s: "",
                                            sleep_seconds=0.0,
                                            max_iterations=1,
                                            signal_threshold=0.5,
                                        )
    mock_submit.assert_called_once_with("AAPL", 5, "buy", 100.0, 0.01, 0.02)


def test_run_trade_loop_skips_when_position_open():
    with patch.object(rl, "SentimentAnalyzer", return_value=DummySentiment()):
        with patch.object(rl, "LSTMPricePredictor", DummyModel):
            with patch.object(rl, "torch", DummyTorch()):
                with patch.object(rl.order_submitter, "submit_bracket_order") as mock_submit:
                    with patch.object(rl.position_manager, "get_open_position", return_value={"symbol": "AAPL"}):
                        with patch.object(rl.volatility_manager, "compute_stop_target", return_value=(0.01, 0.02)):
                            with patch.object(rl.position_sizer, "calculate_position_size", return_value=5):
                                with patch.object(rl, "_fetch_equity", return_value=10000):
                                    with patch.object(rl.time, "sleep", return_value=None):
                                        rl.run_trade_loop(
                                            symbol="AAPL",
                                            seq_len=1,
                                            price_fetcher=lambda s: 100.0,
                                            news_fetcher=lambda s: "",
                                            sleep_seconds=0.0,
                                            max_iterations=1,
                                            signal_threshold=0.5,
                                        )
    mock_submit.assert_not_called()
