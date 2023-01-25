from urllib import request
import json
import base64
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="simple_request",
        description="Performs a POST request with specified by location image"
                    "and shows the response",
    )
    parser.add_argument('-u', '--url', required=True, help="URL to request")
    parser.add_argument('-i', '--image', required=True, help="Path to image to send")
    parser.add_argument('-k', '--json_key', default='classification')

    args = parser.parse_args()
    data = json.dumps({
        args.json_key: base64.b64encode(open(args.image, "rb").read()).decode('ascii')
    }).encode('ascii')

    req = request.Request(args.url, data, {'Content-Type': 'application/json'})
    resp = request.urlopen(req)
    print(resp.read().decode('ascii'))
