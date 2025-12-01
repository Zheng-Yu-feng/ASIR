import asir
import torch
import argparse
from utils import setup_logger

def create_argparser():
    parser = argparse.ArgumentParser(description="Diffusion model argument parser")
    parser.add_argument('--model', type=str, default="google/ddpm-church-256", help="Model name or path")
    parser.add_argument('--num_inference_steps', type=int, default=50, help="Number of inference steps")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument('--device', type=str, default="cuda:1", help="Device to run the model on (e.g., 'cpu', 'cuda:1')")
    parser.add_argument('--channels', type=bool, default=False, help="The choice of embeding channels")
    parser.add_argument('--log_file', type=str, default="sample.log")
    args = parser.parse_args()
    return args

#setting...
args = create_argparser()
logger = setup_logger("my_logger", args.log_file)
logger.info("Logger is set up and running.")
seed = "{}".format(args.seed).encode("utf-8")
logger.info(f"seed setting:{seed}")
ASIR = asir.ASIR(seed=seed, repo=args.model)

# random messages
num_channels = 3 if args.channels else 1
shape = (num_channels, 256, 256) if args.channels else (256, 256)
encode_message = torch.randint(0, 2, shape, device=args.device, dtype=torch.int64)

#embeding...
logger.info(f"message length:{encode_message.numel()}")
alice_generate_results = ASIR.generate(encode_message,args.channels)
last = ASIR.scheduler.num_inference_steps - 1
alice_hidden_sample = alice_generate_results["samples"][last]["hidden"]

# save and load stego image
fname = "stego_image.png"
ASIR.save_sample(alice_hidden_sample, fname)
received_hidden_sample = ASIR.load_sample(fname)

#extracting...
samples_dict = {k: alice_generate_results["samples"][last][k] for k in ASIR.all_keys}
decode_message, errore_rate = ASIR.reveal_initial(samples_dict, received_hidden_sample, encode_message, args.channels)
logger.info(f"initially decode correct rate:{1-errore_rate}")

if args.channels:
    correct_final, message_final = ASIR.update_channels(alice_generate_results,
                                                               decode_message,encode_message,
                                                               received_hidden_sample,
                                                               args.device
                                                               )
else:
    correct_final, message_final = ASIR.update_channel_0(alice_generate_results,
                                                                decode_message,
                                                                encode_message,
                                                                received_hidden_sample,
                                                                args.device)

logger.info(f"finally decode correct rate:{correct_final}")