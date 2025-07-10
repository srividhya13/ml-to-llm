class Solution:

    def get_model_prediction(self, X, weights):
        predictions = []
        for row in X:
            prediction = 0
            for i in range(len(weights)):
                prediction += row[i] * weights[i]
            predictions.append(round(prediction, 5))
        return predictions

    def get_error(self, model_prediction, ground_truth):
        squared_errors = []
        for i in range(len(model_prediction)):
            error = model_prediction[i] - ground_truth[i]
            squared_errors.append(error ** 2)
        mean_error = sum(squared_errors) / len(squared_errors)
        return round(mean_error, 5)
