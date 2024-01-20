
def MetricSelector(args):

    fn_metric = None

    if args.name_metric == 'psnrssim':
        ## description ##

        from Metric.psnrssim import MetricPSNRSSIM
        fn_metric = MetricPSNRSSIM(args.scope, args.batch_size)

    elif args.name_metric == '':
        pass

    return fn_metric