def file_send(file_path):  # 发送大文件可以该方法
    with open(file_path, 'rb') as f:
        while 1:
            data = f.read(20 * 1024 * 1024)  # 每次读取20M
            if not data:
                break
            yield data
