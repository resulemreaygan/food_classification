"""
Author: Resul Emre AYGAN
"""

import os

from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_user, logout_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename as sf

from app_flask import db
from dl_models import Operations
from flask_models import User

auth = Blueprint('auth', __name__)
op = Operations()


@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        email = request.form.get('email')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False
        user = User.query.filter_by(email=email).first()

        if not user:
            flash('Lütfen önce kaydolun!')

            return redirect(url_for('auth.signup'))

        elif not check_password_hash(user.password, password):
            flash('Lütfen giriş bilgilerinizi kontrol edin ve tekrar deneyin.')

            return redirect(url_for('auth.login'))

        login_user(user, remember=remember)

        return redirect(url_for('main.detect'))


@auth.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'GET':
        return render_template('signup.html')

    else:
        email = request.form.get('email')
        name = request.form.get('name')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()

        if user:
            flash('E-posta adresi zaten kayıtlı!')
            return redirect(url_for('auth.signup'))

        new_user = User(email=email, name=name, password=generate_password_hash(password, method='sha256'))

        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('auth.login'))


@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.index'))


@auth.route('/predict', methods=['POST'])
@login_required
def predict_image():
    global secure_filename

    if request.method == "POST":
        img_file = request.files["image_file"]
        secure_filename = sf(img_file.filename)
        img_path = secure_filename
        img_file.save(img_path)

        result = op.predict_image(image_path=img_path, model_path=op.all_const.weight_path)

        os.remove(img_path)

        print("Görüntü Başarıyla Yüklendi")

        result_str = str(result[0][0][0]) + " : " + str(result[0][0][1])
        flash(result_str)
        return redirect(url_for('main.detect'))

    return "Görüntü Yükleme Hata Oluştu!"
