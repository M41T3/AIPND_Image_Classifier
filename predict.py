import predict_utils
import json

in_arg = predict_utils.get_cmd_args() # get cmd args

# load model from checkpoint
model,_,_ =predict_utils.load_model(in_arg.checkpoint, "cuda:0" if in_arg.gpu else "cpu")

# get topk probabilities
topk_prob_array, topk_classes = predict_utils.predict(in_arg.input, model, in_arg.topk)

print("Results:")
if not in_arg.category_names == None:   # if class to category name is available

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    cat_to_name_dict = dict()
    for key in cat_to_name: #
        cat_to_name_dict[int(key)] = cat_to_name[key]

    topk_classes = [int(item) for item in topk_classes]
    flower_name = [cat_to_name_dict[i] for i in topk_classes]

    for i in range(len(topk_prob_array)):
        print("{}. {}\t{}%".format(i+1, flower_name[i].title(),
                                                topk_prob_array[i]*100))
else:
    for i in range(len(topk_prob_array)):
        print("{}. Class: {}\t{}%".format(i+1, topk_classes[i],
                                                topk_prob_array[i]*100))
