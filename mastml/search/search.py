# use: python -m masmtl.grid_search settings.conf data.csv -o results/

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAterials Science Toolkit - Machine Learning')

    parser.add_argument('conf_path', type=str, help='path to mastml .conf file')
    parser.add_argument('data_path', type=str, help='path to csv or xlsx file')
    parser.add_argument('-o', action="store", dest='outdir', default='results',
                        help='Folder path to save output files to. Defaults to results/')

    args = parser.parse_args()

    check_paths(os.path.abspath(args.conf_path),
                os.path.abspath(args.data_path),
                os.path.abspath(args.outdir))

    utils.activate_logging(args.outdir, args)


    try:
        mastml_run(os.path.abspath(args.conf_path),
                   os.path.abspath(args.data_path),
                   os.path.abspath(args.outdir))
    except utils.MastError as e:
        # catch user errors, log and print, but don't raise and show them that nasty stack
        log.error(str(e))
    except Exception as e:
        # catch the error, save it to file, then raise it back up
        log.error('A runtime exception has occured, please go to '
                      'https://github.com/uw-cmg/MAST-ML/issues and post your issue.')
        log.exception(e)
        raise e

