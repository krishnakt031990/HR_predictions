__author__ = 'krishnateja'

import numpy as np
from flask import Flask, abort, jsonify, request
import cPickle as pickle

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)


@app.route('/api', methods=['POST'])
def make_predict():
    data = request.get_json(force=True)
    predict_request = [data['satisfaction_level'], data['last_evaluation'], data['number_project'],
                       data['average_montly_hours'], data['time_spend_company'], data['Work_accident'],
                       data['promotion_last_5years'], data['department'], data['salary']]
    predict_requests = np.array(predict_request)

    y_hat = loaded_model.predict(predict_requests)
    output = [y_hat[0]]
    return jsonify(results=output)


if __name__ == "__main__":
    app.run(port=9000, debug=True)
