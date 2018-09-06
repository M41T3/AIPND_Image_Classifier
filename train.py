import classifier_utils

def main():
    in_arg = classifier_utils.get_cmd_args()
    print(in_arg.data_dir)
    
    dataloaders, class_to_idx = classifier_utils.transform_data(in_arg.data_dir)
    print(class_to_idx)
    
if __name__ == "__main__":
    main()