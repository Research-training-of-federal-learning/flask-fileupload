# coding=UTF8
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
from werkzeug.datastructures import FileStorage
import os
from flask_bootstrap import Bootstrap
import FLAME
import torch
import shutil

from file import file_exists
from file import file_read
from file import del_file

from train import train  # 正确率
from training import training  # 训练三
from training_GTSRB import training_GTSRB
#from training_PUBFIG import training_PUBFIG
from training_freestyle import training_freestyle
from fixtraining import fixtraining  # 训练

from reverse.MNIST import find_lead
from reverse.MNIST import data_statistics
from reverse.MNIST import outputhtml
from reverse.MNIST import find_point
from reverse.MNIST import last_re

from reverse.GTSRB import GTSRB_find_lead
from reverse.GTSRB import GTSRB_data_statistics
from reverse.GTSRB import GTSRB_outputhtml
from reverse.GTSRB import GTSRB_find_point
from reverse.GTSRB import GTSRB_last_re

from reverse.PUBFIG import PUBFIG_find_lead
from reverse.PUBFIG import PUBFIG_data_statistics
from reverse.PUBFIG import PUBFIG_outputhtml
from reverse.PUBFIG import PUBFIG_find_point
from reverse.PUBFIG import PUBFIG_last_re

#新增 neaural cleanse
from NeuralCleanse import train_GTSRB
from NeuralCleanse import train_MNIST
from NeuralCleanse import train_PUBFIG

#新增 MESA
#from MESA import mymain
#from MESA import mymain_MNIST
#from MESA import mymain_pubfig

#新增后门攻击
from Attack import attack_training_MNIST
from Attack import attack_training_GTSRB
from Attack import attack_training_PUBFIG

from choose_dataset_model import choose_datasets
from choose_dataset_model import choose_models
from choose_dataset_model import freestyle_dataset
from choose_dataset_model import freestyle_model

basedir = os.path.abspath(os.path.dirname(__file__))#__file__是Python内置的变量，它包含当前模块的路径和文件名

app = Flask(__name__,static_folder= os.getcwd() + '/static',template_folder=os.getcwd() + '/templates')#__name__是一个特殊变量，用于表示当前模块的名称。如果一个模块被直接执行，那么它的__name__值为__main__；如果一个模块被导入到其他模块中使用，那么它的__name__值为该模块的名称
app.config["SECRET_KEY"] = 'TPmi4aLWRbyVq8zu9v82dWYW1'
bootstrap = Bootstrap(app)

from logging import Formatter, FileHandler

handler = FileHandler(os.path.join(basedir, 'log.txt'), encoding='utf8')#日志处理器
handler.setFormatter(
    Formatter("[%(asctime)s] %(levelname)-8s %(message)s", "%Y-%m-%d %H:%M:%S")
)
app.logger.addHandler(handler)

# app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','tar'])


def allowed_file(filename):#检查一个文件名是否符合应用程序允许上传的文件类型
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.context_processor
def override_url_for():#添加时间戳，浏览器可以快速识别文件是否已更新，并在必要时重新加载文件。
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
def css_static(filename):#当请求到达这个路由时，Flask会调用css_static函数来处理请求，并把请求路径中的filename传入该函数。
    return send_from_directory(app.root_path + '/static/css/', filename)


@app.route('/js/<path:filename>')
def js_static(filename):
    return send_from_directory(app.root_path + '/static/js/', filename)


@app.route('/')
def index():#当用户访问应用的根目录时，该函数会被执行。函数体内的代码使用 redirect 函数将请求重定向到 login 路由。
    return redirect(url_for('login'))
    #url_for 生成指定路由的 URL 地址


@app.route('/logout')
def out():
    if session['is_login'] == False:#如果用户没有登录，将会返回主页
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
    file_path = "pre_models/" + database + "/" + model
    # file_name = request.form.get('file_name')#不同模式
    # file_path = request.form.get('file_path')#不同模式
    return send_from_directory(file_path, file_name, as_attachment=True)#as_attachment=True则表示文件将以附件的形式下载

@app.route("/download2", methods=['GET', 'POST'])
def download2():
    # database = request.form.get('database')
    # model = request.form.get('model')
    # file_name = "pre_model.pth"
    # file_path = "pre_models/" + database + "/" + model
    # file_name = request.form.get('file_name')#不同模式
    # file_path = request.form.get('file_path')#不同模式
    return send_from_directory("choose_dataset_model/", "自定义脚本.zip", as_attachment=True)#as_attachment=True则表示文件将以附件的形式下载



@app.route("/pic_view1", methods=['GET', 'POST'])
def pic_view1():
    return render_template('pic_view1.html')

@app.route("/pic_view", methods=['GET', 'POST'])
def pic_view():
    return render_template('pic_view.html')

@app.route("/view1", methods=['GET', 'POST'])
def view1():
    return render_template('view1.html')

@app.route("/result1", methods=['GET', 'POST'])
def result1():
    return render_template('result1.html')


@app.route("/result2", methods=['GET', 'POST'])
def result2():
    return send_from_directory('templates', 'example.png')

@app.route("/result3", methods=['GET', 'POST'])
def result3():
    return send_from_directory('templates', 're.png')

@app.route("/result4", methods=['GET', 'POST'])
def result4():
    return render_template('result4.html')

@app.route("/Trust_degree", methods=['GET', 'POST'])
def Trust_degree():
    return render_template('Trust_degree.html')

@app.route("/resultNCtrigger", methods=['GET', 'POST'])
def resultNCtrigger():
    return send_from_directory('NeuralCleanse/output', 'trigger.png')

@app.route("/resultNCmask", methods=['GET', 'POST'])
def resultNCmask():
    return send_from_directory('NeuralCleanse/output', 'mask.png')

@app.route("/resultNCnorm", methods=['GET', 'POST'])
def resultNCnorm():
    return send_from_directory('NeuralCleanse/output', 'normDistribution.png')

@app.route("/resultMESAtrigger", methods=['GET', 'POST'])
def resultMESAtrigger():
    return send_from_directory('MESA/output', 'trigger.png')

@app.route("/resultMESAASR", methods=['GET', 'POST'])
def resultMESAASR():
    return send_from_directory('MESA/output', 'ASRDistribution.png')

@app.route('/downloadG0')
def downloadG0():
    current_path = "FLAME_models"
    return send_from_directory(file_path, file_name, as_attachment=True)


@app.route('/project', methods=['GET', 'POST'])
def project():
    if request.method == 'POST':
        m = request.form.get('m')  # 不同模式
        database = request.form.get('database')
        model = request.form.get('model')
        #print(m)
        if (m == '1'):  # 检测历史训练数据
            # 检测是否存在预训练
            if (file_exists.file_exists(
                    "pre_models/" + database + "/" + model + "/pre_model.pth") and file_exists.file_exists(
                    "pre_models/" + database + "/" + model + "/acc.txt")):
                pre = "检测到历史训练数据,"
                acc = file_read.file_read("pre_models/" + database + "/" + model + "/acc.txt")
                addr = ""
                return render_template('project.html', flask_database=database, flask_model=model, flask_pre=pre,
                                       flask_acc=acc, flask_model_download=addr, result1_text="")#这些参数将在模板中被使用，用于动态地生成 HTML 页面


            else:
                if(os.path.exists("pre_models/" + database)):
                    pass
                else:
                    os.mkdir("pre_models/" + database)
                if(os.path.exists("pre_models/" + database + "/" + model)):
                    pass
                else:
                    os.mkdir("pre_models/" + database + "/" + model)
                pre = "未检测到历史训练数据,"
                return render_template('project.html', flask_database=database, flask_model=model, flask_pre=pre,
                                       result1_text="")
        elif (m == '2'):  # 进行正常训练
            if (database == "mnist" and model == "simplenet"):
                # acc = train.main("train/configs/mnist_params.yaml","mnist")
                acc, model_name = training.main("train/configs/mnist_params.yaml", "mnist")
            elif (database == "GTSRB" and model == "6Conv+2Dense"):
                acc, model_name = training.main("training/configs/gtsrb_params.yaml", "GTSRB")
            elif (database == "PUBFIG" and model == "vgg16"):
                acc, model_name = training.main("training/configs/pubfig_params.yaml", "PUBFIG")
            else:
                if(database == "自定义"):
                    train_dataset, test_dataset=freestyle_dataset.freestyledataset()
                else:
                    train_dataset, test_dataset=choose_datasets.choosedatasets(database,".data")
                if(model == "自定义"):
                    model=freestyle_model.freestylemodel()
                else:
                    model=choose_models(model)
                acc, model_name = training_freestyle.main("training/configs/pubfig_params.yaml", database,train_dataset, test_dataset,model)


            shutil.move(model_name, "pre_models/" + database + "/" + model)
            if (file_exists.file_exists("pre_models/" + database + "/" + model + "/pre_model.pth")):
                os.remove("pre_models/" + database + "/" + model + "/pre_model.pth")  # 删除原预训练文件
            # shutil.rmtree("training/saved_models")
            # os.mkdir("training/saved_models")
            os.rename("pre_models/" + database + "/" + model + "/model_last.pt.tar",
                      "pre_models/" + database + "/" + model + "/pre_model.pth")
            f = open("pre_models/" + database + "/" + model + "/acc.txt", "w")
            f.write(acc)
            f.close()
            return render_template('project.html', flask_database=database, flask_model=model, flask_acc=acc,
                                   result1_text="")
        elif (m == '3'):  # 进行逆向检测（输出可能性表单）（疑似嫌犯）（回1）
            # 接收两个参数：轮数，优化率,文件位置
            if (database == "mnist" and model == "simplenet"):
                check_epo = int(request.form.get('check_epo'))
                # check_epo=10
                pretrained_model = "reverse/MNIST/lenet_mnist_model.pth"
                use_cuda = True
                epsilons = [0.05]
                epsilons[0] = float(request.form.get('check_lr'))
                find_result = find_lead.find(check_epo, pretrained_model, use_cuda, epsilons)
                statistics_result_t4, statistics_result_t5 = data_statistics.MNIST_statistics(
                    find_result[check_epo - 1])
                outputhtml.writeHTML("result1.html", statistics_result_t4, statistics_result_t5)
                if (file_exists.file_exists("templates/result1.html")):
                    os.remove("templates/result1.html")  # 删除原预训练文件
                shutil.move("result1.html", "templates/result1.html")

                result1_text = "疑似后门："
                for i in range(len(statistics_result_t4)):
                    for j in range(len(statistics_result_t4[i])):
                        if (statistics_result_t4[i][j] == 1):
                            result1_text += "(" + str(i) + "->" + str(j) + "):" + str(
                                int(statistics_result_t5[i][j])) + "%|"

                return render_template('project.html', flask_database=database, flask_model=model,
                                       result1_text=result1_text)
            elif (database == "GTSRB" and model == "6Conv+2Dense"):
                check_epo = int(request.form.get('check_epo'))
                # check_epo=10
                pretrained_model = "reverse/GTSRB/lenet_mnist_model.pth"
                use_cuda = True
                epsilons = [0.1]
                epsilons[0] = float(request.form.get('check_lr'))
                find_result = GTSRB_find_lead.find(check_epo, pretrained_model, use_cuda, epsilons)
                statistics_result_t4, statistics_result_t5 = GTSRB_data_statistics.PUBFIG_statistics(
                    find_result[check_epo - 1])
                GTSRB_outputhtml.writeHTML("result1.html", statistics_result_t4, statistics_result_t5)
                if (file_exists.file_exists("templates/result1.html")):
                    os.remove("templates/result1.html")  # 删除原预训练文件
                shutil.move("result1.html", "templates/result1.html")

                result1_text = "疑似后门："
                for i in range(len(statistics_result_t4)):
                    for j in range(len(statistics_result_t4[i])):
                        if (statistics_result_t4[i][j] == 1):
                            result1_text += "(" + str(i) + "->" + str(j) + "):" + str(
                                int(statistics_result_t5[i][j])) + "%|"

                return render_template('project.html', flask_database=database, flask_model=model,
                                       result1_text=result1_text)
            elif (database == "PUBFIG" and model == "vgg16"):
                check_epo = int(request.form.get('check_epo'))
                # check_epo=10
                pretrained_model = "reverse/PUBFIG/lenet_mnist_model.pth"
                use_cuda = True
                epsilons = [0.1]
                epsilons[0] = float(request.form.get('check_lr'))
                find_result = PUBFIG_find_lead.find(check_epo, pretrained_model, use_cuda, epsilons)
                statistics_result_t4, statistics_result_t5 = PUBFIG_data_statistics.PUBFIG_statistics(
                    find_result[check_epo - 1])
                GTSRB_outputhtml.writeHTML("result1.html", statistics_result_t4, statistics_result_t5)
                if (file_exists.file_exists("templates/result1.html")):
                    os.remove("templates/result1.html")  # 删除原预训练文件
                shutil.move("result1.html", "templates/result1.html")

                result1_text = "疑似后门："
                for i in range(len(statistics_result_t4)):
                    for j in range(len(statistics_result_t4[i])):
                        if (statistics_result_t4[i][j] == 1):
                            result1_text += "(" + str(i) + "->" + str(j) + "):" + str(
                                int(statistics_result_t5[i][j])) + "%|"

                return render_template('project.html', flask_database=database, flask_model=model,
                                       result1_text=result1_text)



        elif (m == '4'):  # 定位
            if (database == "mnist" and model == "simplenet"):
                # 清空find_result
                if (file_exists.file_exists("reverse/MNIST/find_result")):
                    del_file.del_file("reverse/MNIST/find_result")  # 删除原定位文件
                # 定位
                pretrained_model = "reverse/MNIST/lenet_mnist_model.pth"
                safe_model = "reverse/MNIST/safe_mnist_model.pth"
                # pretrained_model = "safe_mnist_model.pth"
                use_cuda = True
                epsilons = [0.1]  # 作用仅为跑一轮
                num = [[0, 8], [1, 8], [2, 8], [3, 8], [4, 8], [5, 8], [6, 8], [7, 8], [9, 8]]
                mytarget = 8
                # pic_num = 500
                pic_num = 10
                r = True  # 是否继续
                o = False
                find_point.find(num, safe_model, pretrained_model, use_cuda, epsilons, r, o, mytarget, pic_num)
                # 复制结果文件
                if (file_exists.file_exists("reverse/MNIST/find_result_backup")):
                    del_file.del_file("reverse/MNIST/find_result_backup")  # 删除原定位文件
                    os.rmdir("reverse/MNIST/find_result_backup")
                shutil.copytree("reverse/MNIST/find_result", "reverse/MNIST/find_result_backup")
                # 输出定位
                o = True  # 是否输出（本次保存的内容无法用于下一次迭代，但下次迭代会使用上次的结果）
                find_point.find(num, safe_model, pretrained_model, use_cuda, epsilons, r, o, mytarget, pic_num)
                shutil.move("reverse/MNIST/find_result/example.png", "templates")
                return render_template('project.html', flask_database=database, flask_model=model)
            elif (database == "GTSRB" and model == "6Conv+2Dense"):
                # 清空find_result
                if (file_exists.file_exists("reverse/GTSRB/find_result")):
                    del_file.del_file("reverse/GTSRB/find_result")  # 删除原定位文件
                # 定位
                pretrained_model = "reverse/GTSRB/lenet_mnist_model.pth"
                safe_model = "reverse/GTSRB/safe_mnist_model.pth"
                # pretrained_model = "safe_mnist_model.pth"
                use_cuda = True
                epsilons = [0.1]  # 作用仅为跑一轮
                num = [[0, 8], [1, 8], [2, 8], [3, 8], [4, 8], [5, 8], [6, 8], [7, 8], [9, 8],[10, 8], [11, 8], [12, 8], [13, 8], [14, 8], [15, 8], [16, 8], [17, 8], [18, 8], [19, 8],[20, 8], [21, 8], [22, 8], [23, 8], [24, 8], [25, 8], [26, 8], [27, 8], [28, 8], [29, 8],[30, 8], [31, 8], [32, 8], [33, 8], [34, 8], [35, 8], [36, 8], [37, 8], [38, 8], [39, 8], [40, 8], [41, 8], [42, 8]]
                mytarget = 8
                # pic_num = 500
                pic_num = 10
                r = True  # 是否继续
                o = False
                GTSRB_find_point.find(num, safe_model, pretrained_model, use_cuda, epsilons, r, o, mytarget, pic_num)
                # 复制结果文件
                if (file_exists.file_exists("reverse/GTSRB/find_result_backup")):
                    del_file.del_file("reverse/GTSRB/find_result_backup")  # 删除原定位文件
                    os.rmdir("reverse/GTSRB/find_result_backup")
                shutil.copytree("reverse/GTSRB/find_result", "reverse/GTSRB/find_result_backup")
                # 输出定位
                o = True  # 是否输出（本次保存的内容无法用于下一次迭代，但下次迭代会使用上次的结果）
                GTSRB_find_point.find(num, safe_model, pretrained_model, use_cuda, epsilons, r, o, mytarget, pic_num)
                shutil.move("reverse/GTSRB/find_result/example.png", "templates")
                return render_template('project.html', flask_database=database, flask_model=model)
            elif (database == "PUBFIG" and model == "vgg16"):
                # 清空find_result
                if (file_exists.file_exists("reverse/PUBFIG/find_result")):
                    del_file.del_file("reverse/PUBFIG/find_result")  # 删除原定位文件
                # 定位
                pretrained_model = "reverse/PUBFIG/lenet_mnist_model.pth"
                safe_model = "reverse/PUBFIG/safe_mnist_model.pth"
                # pretrained_model = "safe_mnist_model.pth"
                use_cuda = True
                epsilons = [0.1]  # 作用仅为跑一轮
                num = [[0, 8], [1, 8], [2, 8], [3, 8], [4, 8], [5, 8], [6, 8], [7, 8], [9, 8],[10, 8], [11, 8], [12, 8], [13, 8], [14, 8], [15, 8], [16, 8], [17, 8], [18, 8], [19, 8],[20, 8], [21, 8], [22, 8], [23, 8], [24, 8], [25, 8], [26, 8], [27, 8], [28, 8], [29, 8],[30, 8], [31, 8], [32, 8], [33, 8], [34, 8], [35, 8], [36, 8], [37, 8], [38, 8], [39, 8], [40, 8], [41, 8], [42, 8]]
                mytarget = 8
                # pic_num = 500
                pic_num = 10
                r = True  # 是否继续
                o = False
                PUBFIG_find_point.find(num, safe_model, pretrained_model, use_cuda, epsilons, r, o, mytarget, pic_num)
                # 复制结果文件
                if (file_exists.file_exists("reverse/PUBFIG/find_result_backup")):
                    del_file.del_file("reverse/PUBFIG/find_result_backup")  # 删除原定位文件
                    os.rmdir("reverse/PUBFIG/find_result_backup")
                shutil.copytree("reverse/PUBFIG/find_result", "reverse/PUBFIG/find_result_backup")
                # 输出定位
                o = True  # 是否输出（本次保存的内容无法用于下一次迭代，但下次迭代会使用上次的结果）
                PUBFIG_find_point.find(num, safe_model, pretrained_model, use_cuda, epsilons, r, o, mytarget, pic_num)
                shutil.move("reverse/PUBFIG/find_result/example.png", "templates")
                return render_template('project.html', flask_database=database, flask_model=model)


        elif (m == '5'):  # 还原（嫌疑人画像）（上传）
            if (database == "mnist" and model == "simplenet"):
                pretrained_model = "reverse/MNIST/lenet_mnist_model.pth"
                use_cuda=True
                #epsilons = [0, .05, .1, .15, .2, .25, .3]
                epsilons = [.1]
                #epsilons = [0.]
                ep2 = 1
                mytarget = 1
                last_re.find(1,pretrained_model,use_cuda,epsilons,ep2,mytarget)

                # 输出逆向
                shutil.move("reverse/MNIST/re_result/re.png", "templates")
                return render_template('project.html', flask_database=database, flask_model=model)
            elif (database == "GTSRB" and model == "6Conv+2Dense"):
                pretrained_model = "reverse/GTSRB/lenet_mnist_model.pth"
                use_cuda=True
                #epsilons = [0, .05, .1, .15, .2, .25, .3]
                epsilons = [.1]
                #epsilons = [0.]
                ep2 = 1
                mytarget = 8
                GTSRB_last_re.find(1,pretrained_model,use_cuda,epsilons,ep2,mytarget)

                # 输出逆向
                shutil.move("reverse/GTSRB/re_result/re.png", "templates")
                return render_template('project.html', flask_database=database, flask_model=model)
            elif (database == "PUBFIG" and model == "vgg16"):
                pretrained_model = "reverse/PUBFIG/lenet_mnist_model.pth"
                use_cuda=True
                #epsilons = [0, .05, .1, .15, .2, .25, .3]
                epsilons = [.1]
                #epsilons = [0.]
                ep2 = 1
                mytarget = 8
                PUBFIG_last_re.find(1,pretrained_model,use_cuda,epsilons,ep2,mytarget)

                # 输出逆向
                shutil.move("reverse/PUBFIG/re_result/re.png", "templates")
                return render_template('project.html', flask_database=database, flask_model=model)
            pass

        elif (m == '6'):  # 防御性训练（输出下降图表）
            if (database == "mnist" and model == "simplenet"):
                print("fix_models/" + database + "/" + model)
                
            
                # acc = train.main("train/configs/mnist_params.yaml","mnist")
                acc, model_name = fixtraining.main("fixtraining/configs/mnist_params.yaml", "mnist")
                print(model_name)

                shutil.move(model_name, "fix_models/" + database + "/" + model)
                if (file_exists.file_exists("fix_models/" + database + "/" + model + "/fix_model.pth")):
                    os.remove("fix_models/" + database + "/" + model + "/fix_model.pth")  # 删除原预训练文件
            # shutil.rmtree("training/saved_models")
            # os.mkdir("training/saved_models")
                os.rename("fix_models/" + database + "/" + model + "/model_last.pt.tar",
                      "fix_models/" + database + "/" + model + "/fix_model.pth")
                f = open("fix_models/" + database + "/" + model + "/acc.txt", "w")
                f.write(acc)
                f.close()
                return render_template('project.html', flask_database=database, flask_model=model, flask_acc=acc,
                                   result1_text="")
            pass

        elif (m == '7'):  # 安全聚类
            #print(111)
            global G0
            database = request.form.get('database')
            model = request.form.get('model')
            file_path = "pre_models/" + database + "/" + model
            models = FLAME.data_needed(file_path)
            for i in range(len(models)):
                models[i] = file_path+'\\' + models[i]
            n = len(models)  # 参与训练的客户端数量
            size = 10742334  # 模型的参数个数（pubfig就是145002878，可以不用改）
            G0_file_path="FLAME_models\\"+ database + "/" + model+"\\G0.pt"
            fmodle = FLAME.FLAME(n, size, models, G0_file_path)
            fmodle.update()
            fmodle.draw_sinlevel()
            fmodle.draw_sinlevel2()
            fmodle.draw_sinlevel3()
            G0 = fmodle.get_G()  # 重新保存G0
            torch.save(G0, G0_file_path)

        elif (m == '8'):  # neaural cleanse
            nc_epo = int(request.form.get('nc_epo')) # 传入两个参数 epo,lr
            nc_lr = float(request.form.get('nc_lr'))
            if (database == "mnist" and model == "simplenet"):#simplenet:backdoor101自带的mnist模型 绑定mnist
                # print("nc_epo:",nc_epo)
                # print("nc_lr:",nc_lr)
                param = { # 设定训练参数
                    "dataset": "MNIST",
                    "Epochs": nc_epo,
                    "batch_size": 64,
                    "lamda": nc_lr,
                    "num_classes": 10,
                    "image_size": (28, 28)
                }
                print(param)
                train_MNIST.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                train_MNIST.reverse_engineer(param)

            elif (database == "GTSRB" and model == "6Conv+2Dense"):
                param = {
                    "dataset": "GTSRB",
                    "Epochs": nc_epo,
                    "batch_size": 64,
                    "lamda": nc_lr,
                    "num_classes": 43,
                    "image_size": (32, 32)
                }
                print(param)
                train_GTSRB.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                train_GTSRB.reverse_engineer(param)
            elif (database == "PUBFIG" and model == "vgg16"):
                param = {
                    "dataset": "PUBFIG",
                    "Epochs": nc_epo,#1,
                    "batch_size": 1,
                    "lamda": nc_lr,
                    "num_classes": 83,
                    "image_size": (224, 224)
                }
                print(param)
                train_PUBFIG.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                train_PUBFIG.reverse_engineer(param)
            with open('NeuralCleanse/BackdoorLabel/' + database + '_sorted_labels.txt', 'r') as f:
                content = f.read().splitlines()
                nc_sorted_labels = list(map(int, content)) # 将 content 列表中的每个元素转换为整数类型
            with open('NeuralCleanse/BackdoorLabel/' + database + '_sorted_norms.txt', 'r') as f:
                content = f.read().splitlines()
                nc_sorted_norms = list(map(float, content)) # 将 content 列表中的每个元素转换为浮点数类型
            return render_template('project.html', flask_nc_labels=nc_sorted_labels , flask_nc_norms=nc_sorted_norms)

        elif (m == '9'):  # MESA
            mesa_epo = int(request.form.get('mesa_epo')) # 传入三个参数 epo,alpha,beta
            mesa_alpha = float(request.form.get('mesa_alpha'))
            mesa_beta = float(request.form.get('mesa_beta'))
            if (database == "mnist" and model == "simplenet"):#simplenet:backdoor101自带的mnist模型 绑定mnist
                # print("nc_epo:",nc_epo)
                # print("nc_lr:",nc_lr)
                param = {
                    "alpha": mesa_alpha,
                    "beta": mesa_beta,
                    "num_epochs": mesa_epo
                }
                print(param)
                mymain_MNIST.mydataload(param,use_cuda=True,pretrained_model="MESA/MNIST_model_last.pt.tar")
            elif (database == "GTSRB" and model == "6Conv+2Dense"):
                param = {
                    "alpha": mesa_alpha,
                    "beta": mesa_beta,
                    "num_epochs": mesa_epo
                }
                print(param)
                mymain.mydataload(param, use_cuda=True, pretrained_model="MESA/GTSRB_model_last.pt.tar")
            elif (database == "PUBFIG" and model == "vgg16"):
                param = {
                    "alpha": mesa_alpha,
                    "beta": mesa_beta,
                    "num_epochs": mesa_epo
                }
                print(param)
                mymain_pubfig.mydataload(param, use_cuda=True, pretrained_model="MESA/PUBFIG_model_last.pt.tar")
            with open('MESA/BackdoorLabel/' + database + '_sorted_labels.txt', 'r') as f:
                content = f.read().splitlines()
                mesa_sorted_labels = list(map(int, content))
            with open('MESA/BackdoorLabel/' + database + '_sorted_asrs.txt', 'r') as f:
                content = f.read().splitlines()
                mesa_sorted_asrs = list(map(float, content))
            return render_template('project.html', flask_mesa_labels=mesa_sorted_labels, flask_mesa_asrs=mesa_sorted_asrs)

        elif (m == '10'):  # 后门攻击

            if (database == "mnist" and model == "simplenet"):#simplenet:backdoor101自带的mnist模型 绑定mnist
                training_MNIST.main("Attack/configs/mnist_params.yaml", "mnist")

            elif (database == "GTSRB" and model == "6Conv+2Dense"):
                training_GTSRB.main("Attack/configs/gtsrb_params.yaml", "gtsrb")

            elif (database == "PUBFIG" and model == "vgg16"):
                training_PUBFIG.main("Attack/configs/pubfig_params.yaml", "pubfig")
            # 攻击结束后，将Attack中的runs复制到外面的runs中，只保留最新的文件
            source_dir = 'Attack/runs'
            target_dir = 'attack_runs'# 目标文件夹路径

            # 如果目标文件夹已经存在，删除它
            try:
                shutil.rmtree(target_dir)
            except FileNotFoundError:
                pass
            # 复制源文件夹到目标文件夹
            shutil.copytree(source_dir, target_dir)

            target_folder = "attack_runs"
            # 获取目标文件夹中所有子文件夹的名称和修改时间
            subfolders = []
            for dirpath, dirnames, filenames in os.walk(target_folder):
                for dirname in dirnames:
                    full_path = os.path.join(dirpath, dirname)
                    mod_time = os.path.getmtime(full_path)
                    subfolders.append((full_path, mod_time))

            # 按照修改时间对子文件夹进行排序，保留最新的一个文件夹名称，删除其余文件夹
            if subfolders:
                subfolders.sort(key=lambda x: x[1], reverse=True)
                latest_subfolder = subfolders[0][0]
                for folder_path, _ in subfolders[1:]:
                    shutil.rmtree(folder_path)

                # 进入最新的子文件夹，获取该文件夹中所有文件的名称和修改时间
                latest_files = []
                for dirpath, dirnames, filenames in os.walk(latest_subfolder):
                    for filename in filenames:
                        full_path = os.path.join(dirpath, filename)
                        mod_time = os.path.getmtime(full_path)
                        latest_files.append((full_path, mod_time))

                # 按照修改时间对文件进行排序，保留最新的一个文件，删除其余文件
                if latest_files:
                    latest_files.sort(key=lambda x: x[1], reverse=True)
                    latest_file = latest_files[0][0]
                    for file_path, _ in latest_files[1:]:
                        os.remove(file_path)

        # flash(m)
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
            conn.close()#手动关闭数据库连接，这样可以释放资源，同时也可以避免数据丢失或错误

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
