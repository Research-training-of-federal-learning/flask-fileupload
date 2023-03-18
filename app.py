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
import FLAME
import torch
import shutil


from file import file_exists
from file import file_read
from file import del_file

from train import train #正确率
from training import training #训练

from reverse.MNIST import find_lead
from reverse.MNIST import data_statistics
from reverse.MNIST import outputhtml
from reverse.MNIST import find_point

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

@app.route('/logout')
def out():
    if session['is_login'] == False:
        return render_template('/')
    else:
        session['is_login'] = False
        session['name'] = ''
        return redirect('/')

@app.route("/download1", methods=['GET', 'POST'])
def download1():
    database = request.form.get('database')
    model = request.form.get('model')
    file_name = "pre_model.pth"
    file_path = "pre_models/"+database+"/"+model
    #file_name = request.form.get('file_name')#不同模式
    #file_path = request.form.get('file_path')#不同模式
    return send_from_directory(file_path, file_name, as_attachment=True)

@app.route("/result1", methods=['GET', 'POST'])
def result1():
    return render_template('result1.html')

@app.route("/result2", methods=['GET', 'POST'])
def result2():
    return send_from_directory('templates','example.png')

@app.route('/downloadG0')
def downloadG0():
    current_path = "FLAME_models"
    return send_from_directory(file_path, file_name, as_attachment=True)


@app.route('/project', methods=['GET', 'POST'])
def project():
    if request.method == 'POST':
        m = request.form.get('m')#不同模式
        database = request.form.get('database')
        model = request.form.get('model')
        if(m=='1'):#检测历史训练数据
            #检测是否存在预训练
            if(file_exists.file_exists("pre_models/"+database+"/"+model+"/pre_model.pth") and file_exists.file_exists("pre_models/"+database+"/"+model+"/acc.txt")):
                pre="检测到历史训练数据,"
                acc=file_read.file_read("pre_models/"+database+"/"+model+"/acc.txt")
                addr=""
                return render_template('project.html',flask_database=database,flask_model=model,flask_pre=pre,flask_acc=acc,flask_model_download=addr,result1_text="")


            else:
                pre="未检测到历史训练数据,"
                return render_template('project.html',flask_database=database,flask_model=model,flask_pre=pre,result1_text="")
        elif(m=='2'):#进行正常训练
            if(database=="mnist" and model=="simplenet"):
                #acc = train.main("train/configs/mnist_params.yaml","mnist")
                acc,model_name = training.main("train/configs/mnist_params.yaml","mnist")

            shutil.move(model_name,"pre_models/"+database+"/"+model)
            if(file_exists.file_exists("pre_models/"+database+"/"+model+"/pre_model.pth")):
                os.remove("pre_models/"+database+"/"+model+"/pre_model.pth") #删除原预训练文件
            #shutil.rmtree("training/saved_models")
            #os.mkdir("training/saved_models")
            os.rename("pre_models/"+database+"/"+model+"/model_last.pt.tar","pre_models/"+database+"/"+model+"/pre_model.pth")
            f = open("pre_models/"+database+"/"+model+"/acc.txt", "w")
            f.write(acc)
            f.close()
            return render_template('project.html',flask_database=database,flask_model=model,flask_acc=acc,result1_text="")
        elif(m=='3'):#进行逆向检测（输出可能性表单）（疑似嫌犯）（回1）
            #接收两个参数：轮数，优化率,文件位置
            if(database=="mnist" and model=="simplenet"):
                check_epo = int(request.form.get('check_epo'))
                #check_epo=10
                pretrained_model = "reverse/MNIST/lenet_mnist_model.pth"
                use_cuda=True
                epsilons = [0.05]
                epsilons[0] = float(request.form.get('check_lr'))
                find_result=find_lead.find(check_epo,pretrained_model,use_cuda,epsilons)
                statistics_result_t4,statistics_result_t5=data_statistics.MNIST_statistics(find_result[check_epo-1])
                outputhtml.writeHTML("result1.html",statistics_result_t4,statistics_result_t5)
                if(file_exists.file_exists("templates/result1.html")):
                    os.remove("templates/result1.html") #删除原预训练文件
                shutil.move("result1.html","templates/result1.html")
                
                result1_text="疑似后门："
                for i in range(len(statistics_result_t4)):
                    for j in range(len(statistics_result_t4[i])):
                        if(statistics_result_t4[i][j]==1):
                            result1_text+="("+str(i)+"->"+str(j)+"):"+str(int(statistics_result_t5[i][j]))+"%|"

                return render_template('project.html',flask_database=database,flask_model=model,result1_text=result1_text)



        elif(m=='4'):#定位
            if(database=="mnist" and model=="simplenet"):
                #清空find_result
                if(file_exists.file_exists("reverse/MNIST/find_result")):
                    del_file.del_file("reverse/MNIST/find_result") #删除原定位文件
                #定位
                pretrained_model = "reverse/MNIST/lenet_mnist_model.pth"
                safe_model = "reverse/MNIST/safe_mnist_model.pth"
                #pretrained_model = "safe_mnist_model.pth"
                use_cuda=True
                epsilons = [0.1] #作用仅为跑一轮
                num = [[0,8],[1,8],[2,8],[3,8],[4,8],[5,8],[6,8],[7,8],[9,8]]
                mytarget = 8
                #pic_num = 500
                pic_num = 10
                r = True #是否继续
                o = False
                find_point.find(num,safe_model,pretrained_model,use_cuda,epsilons,r,o,mytarget,pic_num)
                #复制结果文件
                if(file_exists.file_exists("reverse/MNIST/find_result_backup")):
                    del_file.del_file("reverse/MNIST/find_result_backup") #删除原定位文件
                    os.rmdir("reverse/MNIST/find_result_backup")
                shutil.copytree("reverse/MNIST/find_result","reverse/MNIST/find_result_backup")
                #输出定位
                o = True #是否输出（本次保存的内容无法用于下一次迭代，但下次迭代会使用上次的结果）
                find_point.find(num,safe_model,pretrained_model,use_cuda,epsilons,r,o,mytarget,pic_num)
                shutil.move("reverse/MNIST/find_result/example.png","templates")
                return render_template('project.html',flask_database=database,flask_model=model)

        elif(m=='5'):#还原（嫌疑人画像）（上传）
            pass

        elif(m=='6'):#防御性训练（输出下降图表）
            pass

        elif(m=='7'):#安全聚类
            global G0
            models = FLAME.data_needed("model\\")
            for i in range(len(models)):
                models[i] = "model\\" + models[i]
            n = 5  # 参与训练的客户端数量
            size = 145002878  # 模型的参数个数（pubfig就是145002878，可以不用改）
            G0 = FLAME.torchfile2np("G0.pt")  # G0是上一轮训练出来的模型，我暂时放在全局变量里了
            fmodle = FLAME.FLAME(n, size, models, G0)  # database我暂且理解为收集的客户端模型，每个模型都是保存到本地的实体文件字符串，下标[i]检索
            fmodle.update()
            G0 = fmodle.get_G()  # 重新保存G0
            torch.save(G0,"FLAME_models\\G0.pt")


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
            return redirect('/project')
        else:
            flash('用户名或密码错误！')
            return render_template('login.html')
    return render_template('login.html')

@app.route('/uploadajax', methods=['POST'])
def upldfile():
    if request.method == 'POST':
        try:
            uploadfiles = request.files['upload']
        except KeyError:
            flash('未选择文件')
            return redirect('/project')
        finally:
            if uploadfiles and allowed_file(uploadfiles.filename):
                filename = secure_filename(uploadfiles.filename)
                app.logger.info('FileName: ' + filename)
                updir = os.path.join(basedir, 'upload/')
                uploadfiles.save(os.path.join(updir, filename))
                file_size = os.path.getsize(os.path.join(updir, filename))
                return jsonify(name=filename, size=file_size)
            else:
                flash('未选择文件')
                return redirect('/project')
    else:
        return redirect('/project')

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
