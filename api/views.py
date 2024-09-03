import os
from flask import (
                        Blueprint,
                        current_app,
                        flash,
                        redirect,
                        render_template,
                        request,
                        url_for,
                    )
import utils
from middleware import model_predict

router = Blueprint("app_router", __name__, template_folder="templates")


@router.route("/display/<filename>")
def display_image(filename):
    """
    Display uploaded image in our UI.
    """
    return redirect(
        url_for("static", filename="uploads/" + filename),
        code=301
    )


@router.route('/')
def index():
    return render_template('index.html')


@router.route('/by_product')
def classifier_by_product():
    return render_template('classifier_by_product.html')


@router.route('/by_list')
def classifier_by_list():
    return render_template('classifier_by_list.html')


@router.route('/your_products')
def your_products():
    return render_template('your_products.html')


@router.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        name = request.form['name']
        description = request.form['description']
        file = request.files.get('image')
        if file:
            if file.filename == "":
                flash("No image selected for uploading")
                return redirect(request.url)
            filename = file.filename
            if file and utils.allowed_file(filename):
                file_hash = utils.get_file_hash(file)
                dst_filepath = os.path.join(
                    current_app.config["UPLOAD_FOLDER"],
                    file_hash
                )
                if not os.path.exists(dst_filepath):
                    file.save(dst_filepath)
                flash("Image successfully uploaded and displayed below")

        categories = model_predict(file_hash, name, description)
        categories_str = utils.get_str_categories(categories)
        return redirect(url_for(
            'app_router.result',
            name=name,
            description=description,
            image_file_name=file_hash,
            categories_str=categories_str
            ))
    return redirect(url_for('app_router.index'))


@router.route('/result')
def result():
    name = request.args.get('name')
    description = request.args.get('description')
    image_file_name = request.args.get('image_file_name')
    categories_str = request.args.get('categories_str')
    return render_template(
        'result.html',
        name=name,
        description=description,
        image_file_name=image_file_name,
        categories_str=categories_str
    )
