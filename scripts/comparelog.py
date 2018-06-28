import argparse
import pandas as pd
import h5py
import json
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logfiles', type=str, nargs='+')
    parser.add_argument('--fields', type=str, default='trueret') #,avglen,ent,kl,vf_r2,vf_kl,tdvf_r2,rloss,racc')
    parser.add_argument('--noplot', action='store_true')
    parser.add_argument('--plotfile', type=str, default=None)
    parser.add_argument('--range_end', type=int, default=None)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    assert len(set(args.logfiles)) == len(args.logfiles), 'Log files must be unique'

    fields = args.fields.split(',')

    # Load logs from all files
    fname2log = []
    for fname in args.logfiles:
        with pd.HDFStore(fname, 'r') as f:
            #assert fname not in fname2log
            df = f['log']
            df.set_index('iter', inplace=True)
            fname2log.append(df.loc[:args.range_end, fields])


    # Print stuff
    if args.verbose or not args.noplot or args.plotfile is not None:
        import matplotlib
        if args.plotfile is not None:
            matplotlib.use('Agg')

        import matplotlib.pyplot as plt; plt.style.use('ggplot')
        colorlist = ['red','blue','green','cyan','grey','yellow','orange','hotpink']
        #colorlist = ['tomato','steelblue','sienna',
        ax = None
        i=0
        for (df, fname) in zip(fname2log, args.logfiles):
            with pd.option_context('display.max_rows', 9999):
                print fname, ', color:', colorlist[i]
                print df[-1:]

            #df['vf_r2'] = np.maximum(0,df['vf_r2'])
            if args.verbose:
                print df

            if ax is None:
                ax = df.plot(subplots=True, color=colorlist[i], legend=False, linewidth=0.4)
                #leg = ax.legend()
            else:
                df.plot(subplots=True, ax=ax, color=colorlist[i], legend=False, linewidth=0.4)
            i += 1

        if not args.noplot:
            plt.style.use('seaborn-dark-palette')
            #plt.ylim(ymax=0, ymin=-1000)
            plt.show()
            plt.figure(figsize=(80,20), dpi=400)

        if args.plotfile is not None:
            rect = 0,0,20,80
            plt.savefig(args.plotfile, bbox_inches='tight', dpi=400)


if __name__ == '__main__':
    main()
