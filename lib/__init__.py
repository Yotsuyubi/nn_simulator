from .train import LitNNSimulator


def nn_simulator(ckpt, freq_list_txt="freq.txt"):
    return LitNNSimulator.load_from_checkpoint(
        ckpt, freq_list_txt=freq_list_txt
    )
