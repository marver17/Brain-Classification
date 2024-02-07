from flask import Flask,request, jsonify
import json
app = Flask(__name__)

@app.route('/config', methods=['GET', 'PUT'])


def config():
    with open('config.json') as f:
        config_data = json.load(f)
    if request.method == 'GET':
        return jsonify(config_data)
    elif request.method == 'PUT':
        new_config_data = request.get_json()


        with open('config.json', 'w') as f:
            json.dump(new_config_data, f, indent=2)

        return jsonify({'status': 'success', 'message': 'Configuration updated successfully'})

if __name__ == '__main__':
    app.run(debug=True)