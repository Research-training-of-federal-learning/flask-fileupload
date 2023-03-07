from flask import (
    Flask,
    request,
    render_template,
    send_from_directory,
    url_for,
    jsonify,
    redirect,
    flash,
    session,
    send_file
)
import sqlite3
from function import hash_code
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import os
from flask_bootstrap import Bootstrap


from file import file_exists
from file import file_read

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config["SECRET_KEY"] = 'TPmi4aLWRbyVq8zu9v82dWYW1'
bootstrap = Bootstrap(app)

from logging import Formatter, FileHandler
handler = FileHandler(os.path.join(basedir, 'log.txt'), encoding='utf8')
handler.setFormatter(
    Formatter("[%(asctime)s] %(levelname)-8s %(message)s", "%Y-%m-%d %H:%M:%S")
)
app.logger.addHandler(handler)


app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)


def dated_url_for(endpoint, **values):
    if endpoint == 'js_static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                    'static/js', filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    elif endpoint == 'css_static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                    'static/css', filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)


@app.route('/css/<path:filename>')
def css_static(filename):
    return send_from_directory(app.root_path + '/static/css/', filename)


@app.route('/js/<path:filename>')
def js_static(filename):
    return send_from_directory(app.root_path + '/static/js/', filename)


@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route("/download1", methods=['GET', 'POST'])
def download1():
    database = request.form.get('database')
    model = request.form.get('model')
    file_name = "pre_model.pth"
    file_path = "pre_models/"+database+"/"+model
    #file_name = request.form.get('file_name')#不同模式
    #file_path = request.form.get('file_path')#不同模式
    return send_from_directory(file_path, file_name, as_attachment=True)


@app.route('/project', methods=['GET', 'POST'])
def project():
    if request.method == 'POST':
        m = request.form.get('m')#不同模式
        print(type(m))
        database = request.form.get('database')
        model = request.form.get('model')
        print(database)
        print(model)
        if(m=='1'):#检测历史训练数据
            #检测是否存在预训练
            if(file_exists.file_exists("pre_models/"+database+"/"+model+"/pre_model.pth") and file_exists.file_exists("pre_models/"+database+"/"+model+"/acc.txt")):
                pre="检测到历史训练数据,"
                print(pre)
                acc=file_read.file_read("pre_models/"+database+"/"+model+"/acc.txt")
                addr=""
                return render_template('project.html',flask_database=database,flask_model=model,flask_pre=pre,flask_acc=acc,flask_model_download=addr)


            else:
                pre="未检测到历史训练数据,"
                print(pre)
                return render_template('project.html',flask_database=database,flask_model=model,flask_pre=pre)
        elif(m=='2'):#进行正常训练
            
            return render_template('project.html',flask_database=database,flask_model=model,pre_data=pre_data)
        elif(m=='3'):#进行逆向检测（输出可能性表单）（回1）
            pass

        elif(m=='4'):#定位、还原操作（输出定位、还原样例）（上传）
            pass

        elif(m=='5'):#防御性训练（输出下降图表）
            pass

        elif(m=='6'):#聚类检测
            pass
        elif(m=='7'):#聚类防御
            pass


        #flash(m)
    return render_template('project.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # 获取请求中的数据
        username = request.form.get('username')
        password = hash_code(request.form.get('password'))

        # 连接数据库，判断用户名+密码组合是否匹配
        conn = sqlite3.connect('db.db')
        cur = conn.cursor()
        try:
            # sqlite3支持?占位符，通过绑定变量的查询方式杜绝sql注入
            sql = 'SELECT 1 FROM USER WHERE USERNAME=? AND PASSWORD=?'
            is_valid_user = cur.execute(sql, (username, password)).fetchone()

            # 拼接方式，存在sql注入风险, SQL注入语句：在用户名位置填入 1 or 1=1 --
            # sql = 'SELECT 1 FROM USER WHERE USERNAME=%s AND PASSWORD=%s' % (username, password)
            # print(sql)
            # is_valid_user = cur.execute(sql).fetchone()
        except:
            flash('用户名或密码错误！')
            return render_template('login.html')
        finally:
            conn.close()

        if is_valid_user:
            # 登录成功后存储session信息
            session['is_login'] = True
            session['name'] = username
            return redirect('/')
        else:
            flash('用户名或密码错误！')
            return render_template('login.html')
    return render_template('login.html')


@app.route('/uploadajax', methods=['POST'])
def upldfile():
    if request.method == 'POST':
        files = request.files['file']
        if files and allowed_file(files.filename):
            filename = secure_filename(files.filename)
            app.logger.info('FileName: ' + filename)
            updir = os.path.join(basedir, 'upload/')
            files.save(os.path.join(updir, filename))
            file_size = os.path.getsize(os.path.join(updir, filename))
            return jsonify(name=filename, size=file_size)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password')
        confirm_password = request.form.get('confirm')
        # 判断所有输入都不为空
        if username and password and confirm_password:
            if password != confirm_password:
                flash('两次输入的密码不一致！')
                return render_template('register.html', username=username)
            # 连接数据库
            conn = sqlite3.connect('db.db')
            cur = conn.cursor()
            # 查询输入的用户名是否已经存在
            sql_same_user = 'SELECT 1 FROM USER WHERE USERNAME=?'
            same_user = cur.execute(sql_same_user, (username,)).fetchone()
            if same_user:
                flash('用户名已存在！')
                return render_template('register.html', username=username)
            # 通过检查的数据，插入数据库表中
            sql_insert_user = 'INSERT INTO USER(USERNAME, PASSWORD) VALUES (?,?)'
            cur.execute(sql_insert_user, (username, hash_code(password)))
            conn.commit()
            conn.close()
            # 重定向到登录页面
            return redirect('/login')
        else:
            flash('所有字段都必须输入！')
            if username:
                return render_template('register.html', username=username)
            return render_template('register.html')
    return render_template('register.html')

if __name__ == '__main__':
    app.run(debug=True)
