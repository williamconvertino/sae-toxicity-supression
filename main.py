from src.datasets import EXPERIMENT_DATASET_DICT
from src.models import load_model_and_sae
from src.feature_analysis import run_feature_analysis
from src.classifier_analysis import run_classifier_analysis
from src.model_steering import run_steering_analysis

def main():

    print("\n=== Loading model and SAE ===")
    model, sae = load_model_and_sae()

    for experiment_name, dataset_loader in EXPERIMENT_DATASET_DICT.items():
        
        print(f"\n--- Loading Dataset for: {experiment_name} ---")

        dataset_dict = dataset_loader()

        print(f"\n=== Starting feature analysis for {experiment_name} ===")
    
        analysis_results = run_feature_analysis(
            experiment_name=experiment_name,
            dataset_dict=dataset_dict,
            model=model,
            sae=sae,
            batch_size=8,
            max_length=64,
            k=20
        )

        pos_mean = analysis_results["pos_mean"]
        neg_mean = analysis_results["neg_mean"]

        print(f"\n=== Starting classifier analysis for {experiment_name} ===")

        # run_classifier_analysis(
        #     experiment_name=experiment_name,
        #     dataset_dict=dataset_dict,
        #     model=model,
        #     sae=sae,
        #     pos_mean=pos_mean,
        #     neg_mean=neg_mean,
        #     batch_size=128,
        #     max_length=64,
        # )

        print("\n=== Running model steering ===")

        run_steering_analysis(
            experiment_name,
            model,
            sae,
            pos_mean,
            neg_mean,
            max_new_tokens=80,
        )


    print("\n=== All experiments finished successfully ===")



if __name__ == "__main__":
    main()
