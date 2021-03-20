import argparse
from EVAL.EXPERIMENTS import cobb_exp1_n_3
from EVAL.EXPERIMENTS import cobb_exp2
from EVAL.EXPERIMENTS import convert_probs_to_results


parser = argparse.ArgumentParser()
parser.add_argument('--option', dest='option')
parser.add_argument('--exp', dest='exp')
parser.add_argument('--task', dest='task')
args = parser.parse_args()


if __name__ == '__main__':
    if args.option == 'eval':
        print(f'Evaulating exp={args.exp}, task={args.task}...')
        if args.exp == '1' or args.exp == '3':
            cobb_exp1_n_3.execute(args.exp, args.task)
        else:
            cobb_exp2.execute(args.exp, args.task)

    elif args.option == 'plot':
        print(f'Plotting exp={args.exp}, task={args.task}...')
        convert_probs_to_results.execute(args.exp, args.task)
            


