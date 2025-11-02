from flask_sqlalchemy import SQLAlchemy
import os

db = SQLAlchemy()

class EmotionRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    image_path = db.Column(db.String(200))
    emotion = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    is_online = db.Column(db.Boolean, default=True)