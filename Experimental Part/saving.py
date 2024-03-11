from packages import *

def problem(path_to_file,
            file_name,
            train_meta, test_meta, 
            model:str,
            path_to_model:str,
            training_time,
            inference_time,
            mse,
            evs,
            mae,
            r2,
            y_test_pred,
            y_test_pred_file_name,
            model_parameters):
        
        np.save(os.path.join(path_to_file, y_test_pred_file_name), y_test_pred)

        
        if not os.path.exists(path_to_file):
                os.makedirs(path_to_file)

        if not os.path.exists(path_to_file + file_name):
                my_dict = {"problem": {"split": train_meta["problem"],
                                       "size": train_meta["subsampling_size"],
                                       "subsampling_method": train_meta["subsampling_method"],
                                       "epsilon": train_meta["epsilon"],
                                       "reference_measure": train_meta["reference_measure"]},
                           "sinkhorn": {"train_time": train_meta["execution_time"],
                                        "test_time": test_meta["execution_time"]},
                           "regression": {}}
                my_dict["regression"][model] = {"path_to_model": path_to_model,
                                                "model_parameters": model_parameters,
                                                "mse": mse,
                                                "evs": evs,
                                                "mae": mae,
                                                "r2": r2,
                                                "training_time": training_time,
                                                "inference_time": inference_time,
                                                "y_test_pred": os.path.join(path_to_file, y_test_pred_file_name)}
                joblib.dump(my_dict, path_to_file+file_name)
        else:
                my_dict = joblib.load(path_to_file + file_name)
                my_dict["regression"][model] = {"path_to_model": path_to_model,
                                "model_parameters": model_parameters,
                                "mse": mse,
                                "evs": evs,
                                "mae": mae,
                                "r2": r2,
                                "training_time": training_time,
                                "inference_time": inference_time,
                                "y_test_pred": os.path.join(path_to_file, y_test_pred_file_name)}
                joblib.dump(my_dict, path_to_file+file_name)

        return "Done!"
