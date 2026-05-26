from __future__ import annotations

from flask import Flask, jsonify, request

from .map_ids import load_movie_map
from .svd_recommender import SVDRecommender

app = Flask(__name__)

rec = SVDRecommender()
movie_map = load_movie_map()


@app.route("/recommend", methods=["POST"])
def recommend_post():
    data = request.json or {}
    user = data.get("user_id")
    topk = int(data.get("topk", 10))
    if user is None:
        return jsonify({"error": "send { 'user_id': <id> }"}), 400

    recs = rec.recommend(user, N=topk)
    out = []
    for movie_id, score in recs:
        title = movie_map.get(int(movie_id), "Unknown Title")
        out.append({"movie_id": int(movie_id), "title": title, "score": score})
    return jsonify(out)


@app.route("/recommend/<int:user_id>", methods=["GET"])
def recommend_get(user_id: int):
    recs = rec.recommend(user_id, N=10)
    out = []
    for movie_id, score in recs:
        title = movie_map.get(int(movie_id), "Unknown Title")
        out.append({"movie_id": int(movie_id), "title": title, "score": score})
    return jsonify(out)


@app.route("/", methods=["GET"])
def home():
    return """
    <!doctype html>
    <html>
    <head><title>Movie Recommender Test</title></head>
    <body style="font-family:Arial,Helvetica,sans-serif;padding:20px;">
      <h2>Movie Recommender — Test UI</h2>
      <div>
        <label>User id: <input id="uid" value="1" style="width:60px" /></label>
        <label style="margin-left:10px">Topk: <input id="k" value="5" style="width:40px" /></label>
        <button onclick="call()" style="margin-left:10px">Get Recs</button>
      </div>
      <pre id="out" style="background:#f6f6f6;border:1px solid #ddd;padding:12px;margin-top:12px;"></pre>
      <script>
        async function call(){
          const uid = parseInt(document.getElementById('uid').value);
          const k = parseInt(document.getElementById('k').value);
          const res = await fetch('/recommend', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({user_id: uid, topk: k})
          });
          const data = await res.json();
          document.getElementById('out').textContent = JSON.stringify(data, null, 2);
        }
      </script>
    </body>
    </html>
    """


def main():
    app.run(host="127.0.0.1", port=5000, debug=True)


if __name__ == "__main__":
    main()
