"""
Author: Resul Emre AYGAN
"""

from flask import Blueprint, render_template
from flask_login import login_required, current_user

from app_flask import create_app, db

main = Blueprint('main', __name__)


@main.route('/')
def index():
    return render_template('index.html')


@main.route('/detect')
@login_required
def detect():
    return render_template('detect.html', name=current_user.name)


app = create_app()

if __name__ == '__main__':
    db.create_all(app=create_app())
    app.run(debug=False)
