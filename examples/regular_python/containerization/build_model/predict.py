from mastml.mastml_predictor import make_prediction
import pandas as pd


def main():
    X_train = pd.read_csv('X_train.csv')
    features_to_keep = X_train.columns.tolist()

    X_test = pd.read_csv('/mnt/test.csv')

    pred_dict = make_prediction(
                                X_test='/mnt/test.csv',
                                model='model.pkl',
                                X_test_extra='/mnt/test.csv',
                                preprocessor='preprocessor.pkl',
                                calibration_file='calibration_file.csv',
                                featurize=True,
                                featurizer='ElementalFeatureGenerator',
                                featurize_on='Composition',
                                features_to_keep=features_to_keep,
                                composition_df=X_test['Composition'],
                                )

    pred_dict.to_csv('/mnt/prediction.csv', index=False)

    return pred_dict


if __name__=='__main__':
    pred_dict = main()
