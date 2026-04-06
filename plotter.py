import matplotlib.pyplot as plt

def plotter(args):
    Fulls = ["Training", "Validation"]
    smalls = ["train", "val"]
    if args.question==1:
      sequences = list(range(0,45000,4992))
      for l in range(2):
        plt.figure(figsize=(7, 5))
        for dim in [64,128,256,512]:
          with open(f"{args.loss_folder}/{dim}_{smalls[l]}_losses.txt") as f:
            plt.plot(sequences,[float(line.strip()) for line in f], label=f"dim={dim}")
          plt.title(f'{Fulls[l]} Loss vs. Sequences')
          plt.xlabel('Number of sequences processed')
          plt.ylabel(f'{Fulls[l]} Loss')
          plt.legend()
          plt.grid(True)
          plt.savefig(f"{args.loss_folder}/{smalls[l]}_plot.png")
    elif args.question==2:
      for l in range(2):
        plt.figure(figsize=(7, 5))
        for bs in [32,64,128,256]:
          if bs==128:
            sequences = list(range(0,45000,4992))
          elif bs==256:
            sequences = list(range(0,50000,4864))
          else:
            sequences = list(range(0,50000,4992))
          with open(f"{args.loss_folder}/{bs}_{smalls[l]}_losses.txt") as f:
            plt.plot(sequences,[float(line.strip()) for line in f], label=f"batch size={bs}")
          plt.title(f'{Fulls[l]} Loss vs. Sequences')
          plt.xlabel('Number of sequences processed')
          plt.ylabel(f'{Fulls[l]} Loss')
          plt.legend()
          plt.grid(True)
          plt.savefig(f"{args.loss_folder}/{smalls[l]}_plot.png")
    elif args.question==3:
      for l in range(2):
        plt.figure(figsize=(7, 5))
        sequences = [10000,20000,50000,100000]
        loss = []
        for seq in [10000,20000,50000,100000]:
          with open(f"{args.loss_folder}/{seq}_{smalls[l]}_losses.txt") as f:
            loss.append(float(f.readlines()[-1].strip()))
        plt.plot(sequences,loss)
        plt.title(f'Final {Fulls[l]} Loss vs. Sequences')
        plt.xlabel('Total number of sequences processed')
        plt.ylabel(f'{Fulls[l]} Loss')
        plt.xticks(sequences)
        plt.grid(True)
        plt.savefig(f"{args.loss_folder}/{smalls[l]}_plot.png")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
  
    parser.add_argument("--question", type=int)
    parser.add_argument("--loss_folder", type=str)
    plotter(parser.parse_args())