import pdb

def DataLoaderSelector(args):
    fn_loader = None
    # train dataset
    if args.name_data == 'train':
        ## description ##
        from Dataloader.dataloader_LF import LFDataloader
        fn_loader = LFDataloader(args.dir_data, 
                                 args.x_res, 
                                 args.y_res, 
                                 args.uv_diameter,
                                 args.uv_dilation,
                                 args.data_output_option, 
                                 args.resize_scale, 
                                 args.mode)

    # test, valid datasets
    # elif (args.name_data == 'DeOccNet') or \
    #     (args.name_data == 'DeOccNet_Crop') or \
    #     (args.name_data == 'test_EPFL_10') or \
    #         (args.name_data == 'test_Stanford_Lytro_16') or \
    #         (args.name_data == 'Dense_Quant') or \
    #         (args.name_data == 'Dense_Quant_Double') or \
    #         (args.name_data == 'Mask_Edit'):
    else:
        from Dataloader.dataloader_DeOccNet import DeOccNetDataloader
        fn_loader = DeOccNetDataloader(args.dir_data, 
                                 args.x_res, 
                                 args.y_res, 
                                 args.uv_diameter_image,
                                 args.uv_diameter,
                                 args.uv_dilation,
                                 args.data_output_option, 
                                 args.resize_scale, 
                                 args.mode,
                                 args.specific_dir)
    
    return fn_loader