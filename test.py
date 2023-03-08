# from reverse.MNIST import find_lead
# from reverse.MNIST import data_statistics
# from reverse.MNIST import outputhtml

# pretrained_model = "reverse/MNIST/lenet_mnist_model.pth"
# use_cuda=True
# epsilons = [0.05]
# check_epo=10
# find_result=find_lead.find(check_epo,pretrained_model,use_cuda,epsilons)
# print(find_result[check_epo])
# statistics_result_t4,statistics_result_t5=data_statistics.MNIST_statistics(find_result[check_epo-1])
# outputhtml.writeHTML("template","result1.html",statistics_result_t4,statistics_result_t5)

import os

base_path = os.path.dirname(os.path.realpath("templates/result1"))  # 获取当前路径
print(base_path)