import classifier_utils

def main():
    in_arg = classifier_utils.get_cmd_args()
    print(in_arg.data_dir)
    
    dataloaders, class_to_idx = classifier_utils.transform_data(in_arg.data_dir)
    
    hidden_size = in_arg.hidden_units.split(',')
    hidden_size = [int(x) for x in hidden_size]
    
    model, criterion, optimizer = classifier_utils.create_model(class_to_idx, input_size=25088, hidden_size=hidden_size, output_size=len(class_to_idx))
    print(model)
if __name__ == "__main__":
    main()