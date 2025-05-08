import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import networkx as nx


def findfile(directory, file_prefix):
    filenames=[]
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.startswith(file_prefix):
                filenames.append(fileName)
    return filenames

def findfile_ER(directory, file_prefix, file_suffix):
    filenames=[]
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.startswith(file_prefix) and fileName.endswith(file_suffix):
                filenames.append(fileName)
    return filenames

def bar(network_name):
    if network_name=='ER':
        unit_topics = ["STLD",'MinSum','FINDER','HDA',"HD"]
        dir = "bar_source/"

        file_pre = network_name  # 文件以tes_开头
        fileNames = findfile(dir, file_pre)
        fileNames=fileNames[30:]+fileNames[:30]
        fig, axes = plt.subplots(1, 4,  figsize=(10, 3))
        fig.subplots_adjust(top=0.9, bottom=0.22, left=0.07, right=0.97)

        D=['3','6','9','12']
        color=['#403990',"#80A6E2","#FBDD85","#F46F43","#CF3D3E"]
        num=30
        for j, col in enumerate(axes):
            back_auc = 0
            degree_auc = 0
            finder_auc = 0
            MS_auc = 0
            adpDegree_auc = 0
            ba=[]
            da=[]
            fa=[]
            ma=[]
            aa=[]
            for i in range(j*num,(j+1)*num):
                name = fileNames[i]#[:-4]
                back = np.load('final_DN_result/' + name + '_back.npy')   #读取瓦解曲线的结果
                degree = np.load('final_DN_result/' + name + '_degree.npy')
                finder = np.load('final_DN_result/' +name + '_finder.npy')
                MS = np.load('final_DN_result/' + name + '_MS.npy')
                adpDegree = np.load('final_DN_result/' + name + '_adpDegree.npy')
                max_val=max(back)

                back_auc += back.sum()/(1000*max_val)   #计算平均auc值
                ba.append(back.sum()/(1000*max_val))
                degree_auc += degree.sum()/(1000*max_val)
                da.append(degree.sum()/(1000*max_val))
                finder_auc += finder.sum()/(1000*max_val)
                fa.append(finder.sum()/(1000*max_val))
                MS_auc += MS.sum()/(1000*max_val)
                ma.append(MS.sum()/(1000*max_val))
                adpDegree_auc += adpDegree.sum()/(1000*max_val)
                aa.append(adpDegree.sum()/(1000*max_val))

            std=[np.std(ba, ddof=1),np.std(ma, ddof=1),np.std(fa, ddof=1),np.std(aa, ddof=1),np.std(da, ddof=1)]
            temp=[back_auc/num,MS_auc/num,finder_auc/num,adpDegree_auc/num,degree_auc/num]
            col.bar(unit_topics, temp, yerr = std,error_kw = {'elinewidth':2,'ecolor' : '0.0', 'capsize' :4 },color=color,width=0.75)
            print(std,temp)
            col.set_xticklabels(unit_topics, Rotation=40)
            font1 = {
                     'weight': 'normal',
                     'size': 12,
                     }
            col.set_title(r'$ER_{\bar{D}=' + D[j]+'}$', font1,y=1.0,x=0.6)
        fig.text(0.01, 0.55, 'ANC',va='center', fontsize=12, rotation='vertical')
        plt.savefig("final_result/ER_bar.pdf")
        plt.show()

    if network_name=='SF':
        unit_topics = ['directed', 'undirected']
        dir = "bar_source/"
        file_pre = "SF_1000"
        #dir = "SF_new_lamda2-3.5/"
        #file_pre = "SF_1000"

        fileNames = findfile(dir, file_pre)
        fig, axes = plt.subplots(1, 4, figsize=(10, 3))
        fig.subplots_adjust(top=0.9, bottom=0.22, left=0.07, right=0.97)

        D=['2.2','2.5','2.8','3.2']
        color = ['#403990',  "#00FF00"]
        num = 30
        for j, col in enumerate(axes):
            finder_auc = 0
            finderun_auc = 0
            fa = []
            faun = []


            for i in range(j * num, (j + 1) * num):
                name = fileNames[i]  # [:-4]

                finder_un = np.load('FINDER_directed_un_result/' + name + '_FD_un.npy')
                finder = np.load('FINDER_directed_result/' + name + '_FD.npy')

                max_val = max(finder)

                finder_auc += finder.sum() / (1000 * max_val)
                fa.append(finder.sum() / (1000 * max_val))
                finderun_auc += finder_un.sum() / (1000 * max_val)
                faun.append(finder_un.sum() / (1000 * max_val))



            std = [np.std(fa, ddof=1), np.std(faun, ddof=1)]
            temp = [finder_auc / num, finderun_auc / num]
            bars = col.bar(unit_topics, temp, yerr=std, error_kw={'elinewidth': 2, 'ecolor': '0.0', 'capsize': 4}, color=color,
                    width=0.75)
            # 在每个柱子上方标注数值
            for bar, value in zip(bars, temp):
                height = bar.get_height()
                col.text(bar.get_x() + bar.get_width() / 2, height, f'{value:.5f}', ha='center', va='bottom')
            print(std, temp)
            col.set_xticklabels(unit_topics, rotation=0)
            font1 = {
                     'weight': 'normal',
                     'size': 12,
                     }
            col.set_title(r'$SF_{\lambda=' + D[j]+'}$', font1, y=1.0,x=0.5)
        fig.text(0.01, 0.55, 'ANC', va='center', fontsize=12, rotation='vertical')
        plt.show()



def bar_new():
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.subplots_adjust(top=0.9, bottom=0.05, left=0.07, right=0.97, hspace=0.65)


    dir = "SF_ER_FINDER/ER/"
    file_pre_list = ["ER_100_", "ER_1000_", "ER_10000_"]
    D = ['3.0', '6.0', "9.0", "12.0"]
    color = ['#403990',  "#00FF00"]
    bar_width = 0.15
    index = np.arange(len(D)) * 0.5
    for ep, file_pre in enumerate(file_pre_list):
        num = 30
        data = np.zeros(shape=(len(D), 2))
        data_std = np.zeros(shape=(len(D), 2))
        for j in range(len(D)):
            fileNames = findfile_ER(dir, file_pre,  "_" + D[j])
            finder_auc = 0
            finderun_auc = 0
            fa = []
            faun = []
            print(fileNames)
            for i in range(num):
                name = fileNames[i]  # [:-4]
                g = nx.read_graphml(dir + name)
                print("节点数量", g.number_of_nodes())
                finder_un = np.load('FINDER_directed_un_result/' + name + '_FD_un.npy')
                finder = np.load('FINDER_directed_result/' + name + '_FD.npy')

                max_val = max(finder)
                print(max_val)
                finder_auc += finder.sum() / (g.number_of_nodes() * max_val)
                fa.append(finder.sum() / (g.number_of_nodes() * max_val))
                finderun_auc += finder_un.sum() / (g.number_of_nodes() * max_val)
                faun.append(finder_un.sum() / (g.number_of_nodes() * max_val))
                # finder_auc += finder.sum()
                # fa.append(finder.sum())
                # finderun_auc += finder_un.sum()
                # faun.append(finder_un.sum())

            std = [np.std(fa, ddof=1), np.std(faun, ddof=1)]
            temp = [finder_auc / num, finderun_auc / num]
            print(std, temp)
            data[j, :] = np.array(temp)
            data_std[j, :] = np.array(std)
        axes[0][ep].bar(index, data[:, 0], bar_width, yerr=data_std[:, 0],
               error_kw={'elinewidth': 2, 'ecolor': '0.0', 'capsize': 4}, color=color[0])
        axes[0][ep].bar(index + bar_width, data[:, 1], bar_width, yerr=data_std[:, 1],
               error_kw={'elinewidth': 2, 'ecolor': '0.0', 'capsize': 4}, color=color[1])
        # ax.bar(index, data[:,0], bar_width,error_kw={'elinewidth': 2, 'ecolor': '0.0', 'capsize': 4}, color=color[ep])
        # ax.bar(index + bar_width, data[:,1], bar_width,error_kw={'elinewidth': 2, 'ecolor': '0.0', 'capsize': 4}, color=color[ep], alpha=0.8)
        axes[0][ep].set_xticks(index + bar_width / 2)
        axes[0][ep].set_xticklabels([r"$\langle d \rangle=" + d + '$' for d in D], rotation=0)
        axes[0][ep].set_title(r'$N=' + file_pre[3:len(file_pre)-1] + '$', fontsize =14, y=1.0, x=0.5)
        axes[0][ep].set_ylim(0, 0.5)
        axes[0][ep].tick_params(axis='both', labelsize=12)
    axes[0][0].set_ylabel('ANC', fontsize=14, labelpad=18)
    #axes[0][0].legend(title='Scale-Free Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0][1].legend(['DNetKey', "FINDER"], loc='center', bbox_to_anchor=(0.5, -0.25), ncol=2, borderaxespad=0, fontsize=14)
    axes[0][1].text(0.5, 1.15, r'ER Networks', fontsize=15, ha='center', transform=axes[0][1].transAxes)
    axes[0][0].text(-0.18, 1.14, 'A', transform=axes[0][0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    axes[0][1].text(-0.18, 1.14, 'B', transform=axes[0][1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    axes[0][2].text(-0.18, 1.14, 'C', transform=axes[0][2].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

    dir = "SF_ER_FINDER/SF/"
    file_pre_list = ["SF_100_","SF_1000_","SF_10000_"]
    D = ['2.2','2.4',"2.6","2.8"]
    color = ['#403990',  "#00FF00"]
    bar_width = 0.15
    index = np.arange(len(D))*0.5
    for ep, file_pre in enumerate(file_pre_list):
        num = 30
        data = np.zeros(shape=(len(D), 2))
        data_std = np.zeros(shape=(len(D), 2))
        for j in range(len(D)):
            fileNames = findfile(dir, file_pre +D[j] +"_")
            finder_auc = 0
            finderun_auc = 0
            fa = []
            faun = []

            for i in range(num):
                name = fileNames[i]  # [:-4]
                g = nx.read_graphml(dir + name)
                print("节点数量", g.number_of_nodes())
                finder_un = np.load('FINDER_directed_un_result/' + name + '_FD_un.npy')
                finder = np.load('FINDER_directed_result/' + name + '_FD.npy')
                #finder = np.load('FINDER_directed_un_result/' + name + '_FD_un_1.npy')  #使用无向finder的源码和无向finder模型进行测试
                max_val = max(finder)
                print(max_val)
                finder_auc += finder.sum() / (g.number_of_nodes() * max_val)
                fa.append(finder.sum() / (g.number_of_nodes() * max_val))
                finderun_auc += finder_un.sum() / (g.number_of_nodes() * max_val)
                faun.append(finder_un.sum() / (g.number_of_nodes() * max_val))
                #finder_auc += finder.sum()
                #fa.append(finder.sum())
                #finderun_auc += finder_un.sum()
                #faun.append(finder_un.sum())

            std = [np.std(fa, ddof=1), np.std(faun, ddof=1)]
            temp = [finder_auc / num, finderun_auc / num]
            print(std, temp)
            data[j, :] = np.array(temp)
            data_std[j, :] = np.array(std)
        axes[1][ep].bar(index, data[:,0], bar_width,yerr=data_std[:,0],error_kw={'elinewidth': 2, 'ecolor': '0.0', 'capsize': 4}, color=color[0])
        axes[1][ep].bar(index + bar_width, data[:,1], bar_width,yerr=data_std[:,1],error_kw={'elinewidth': 2, 'ecolor': '0.0', 'capsize': 4}, color=color[1])
        #ax.bar(index, data[:,0], bar_width,error_kw={'elinewidth': 2, 'ecolor': '0.0', 'capsize': 4}, color=color[ep])
        #ax.bar(index + bar_width, data[:,1], bar_width,error_kw={'elinewidth': 2, 'ecolor': '0.0', 'capsize': 4}, color=color[ep], alpha=0.8)
        axes[1][ep].set_xticks(index + bar_width / 2)
        #axes[1][ep].set_xticklabels(D, rotation=0)
        axes[1][ep].set_xticklabels([r"$\lambda=" + d + '$' for d in D], rotation=0)
        axes[1][ep].set_title(r'$N=' + file_pre[3:len(file_pre)-1] + '$', fontsize =14, y=1.0, x=0.5)
        axes[1][ep].set_ylim(0, 0.14)
        axes[1][ep].tick_params(axis='both', labelsize=12)
    axes[1][0].set_ylabel('ANC', fontsize=14)
    #axes[1][0].legend(title='Scale-Free Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1][1].text(0.5, 1.15, r'SF Networks', fontsize=15, ha='center', transform=axes[1][1].transAxes)
    axes[1][0].text(-0.18, 1.14, 'D', transform=axes[1][0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    axes[1][1].text(-0.18, 1.14, 'E', transform=axes[1][1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    axes[1][2].text(-0.18, 1.14, 'F', transform=axes[1][2].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

    # 添加图例到整张图的正中间
    #fig.legend(title='Scale-Free Model', bbox_to_anchor=(0.5, 0.5), loc='center', ncol=1)
    plt.show()


def bar_curve(network_name):
    if network_name=='ER':
        unit_topics = ["DNetKey", "FINDER", "CoreHD", "PageRk", 'MinSum',  "DND", 'HDA', "HD", 'ID','OD']
        dir = "SF_ER_FINDER/ER/"
        # dir = "SF_new/"
        # file_pre = network_name  # 文件以tes_开头
        file_pre = "ER_1000_"



        lamb = [r'$ER_{\langle d \rangle=3}$', r'$ER_{\langle d \rangle=6}$', r'$ER_{\langle d \rangle=9}$', r'$ER_{\langle d \rangle=12}$']

        fig, axes = plt.subplots(2, 4, figsize=(12, 7))
        fig.subplots_adjust(top=0.9, bottom=0.12, left=0.07, right=0.97, hspace=0.45)

        D = ['3.0','6.0','9.0','12.0']
        color = ['#403990', "#00FF00", "#888888",  "#80A6E2", "#FBDD85", "#00FFFF", "#F46F43", "#CF3D3E", "#008000", "#333333"]
        num = 30
        for j, col in enumerate(axes[0]):
            fileNames = findfile_ER(dir, file_pre, "_" + D[j])
            network_names = fileNames  # 调整为参数从小到大的顺序

            FD = np.zeros(1000)  # 初始化不同方法的平均GSCC曲线
            FDun = np.zeros(1000)
            degree = np.zeros(1000)
            adpdegree = np.zeros(1000)
            learn = np.zeros(1000)
            pr = np.zeros(1000)
            dnd = np.zeros(1000)
            corehd = np.zeros(1000)
            degree_in = np.zeros(1000)
            degree_out = np.zeros(1000)

            for epoch in range(num):  # 计算平均GSCC曲线
                network_name = network_names[epoch]
                temp = np.load('FINDER_directed_result/' + network_name + '_FD.npy')
                FD += temp / (num*max(temp))
                temp = np.load('FINDER_directed_un_result/' + network_name + '_FD_un.npy')
                FDun += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_Core.npy')
                corehd += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_PR.npy')
                pr += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_degree.npy')
                degree += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_adpDegree.npy')
                adpdegree += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_MS.npy')
                learn += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_DND.npy')
                dnd += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_ID.npy')
                degree_in += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_OD.npy')
                degree_out += temp / (num*max(temp))

            x = [_ / len(FD) for _ in range(len(FD))]
            # col.set_title(r'AvgD='+D[epoch],y=0.9)
            col.set_title(lamb[j], y=0.8, x=0.5)
            col.plot(x, FD, color='#403990', lw=1.8)  # 绘制平均GSCC曲线
            col.plot(x, FDun, color="#00FF00", lw=1.8)  # 绘制平均GSCC曲线
            col.plot(x, corehd, color="#888888", lw=1.2)
            col.plot(x, pr, color="#80A6E2", lw=1.2)
            col.plot(x, learn, color="#FBDD85", lw=1.2)
            col.plot(x, dnd, color="#00FFFF", lw=1.2)
            col.plot(x, adpdegree, color="#F46F43", lw=1.2)
            col.plot(x, degree, color="#CF3D3E", lw=1.2)
            col.plot(x, degree_in, color="#008000", lw=1.2)
            col.plot(x, degree_out, color="#333333", lw=1.2)
            col.tick_params(labelsize=10)
            col.set_ylim(-0.05, 1.05)

        fig.text(0.5, 0.51, 'Fraction of Nodes Removed', fontsize=12, ha='center')
        fig.text(0.02, 0.75, 'LSCC', va='center', fontsize=12, rotation='vertical')
        col.legend(["DNetKey", "FINDER", "CoreHD", "PageRk", 'MinSum',  "DND", 'HDA', "HD", "ID", "OD"], loc=1, prop={'size': 10},
                   bbox_to_anchor=(0.76, 1.17),  ncol=10, borderaxespad=0, fontsize=12)

        for j, col in enumerate(axes[1]):
            FD_auc = 0
            FDun_auc = 0
            degree_auc = 0
            MS_auc = 0
            adpDegree_auc = 0
            PR_auc = 0
            DND_auc = 0
            Core_auc = 0
            degreein_auc = 0
            degreeout_auc = 0
            fa = []
            fua = []
            da = []
            ma = []
            aa = []
            pa = []
            dnda = []
            corea = []
            dia = []
            doa = []
            fileNames = findfile_ER(dir, file_pre, "_" + D[j])
            for i in range(num):
                name = fileNames[i]  # 调整为参数从小到大的顺序
                FD = np.load('FINDER_directed_result/' + name + '_FD.npy')
                FD_un = np.load('FINDER_directed_un_result/' + name + '_FD_un.npy')
                # print(back)
                degree = np.load('final_DN_result/' + name + '_degree.npy')
                MS = np.load('final_DN_result/' + name + '_MS.npy')
                adpDegree = np.load('final_DN_result/' + name + '_adpDegree.npy')
                PR = np.load('final_DN_result/' + name + '_PR.npy')
                DND = np.load('final_DN_result/' + name + '_DND.npy')
                CoreHD = np.load('final_DN_result/' + name + '_Core.npy')
                degree_in = np.load('final_DN_result/' + name + '_ID.npy')
                degree_out = np.load('final_DN_result/' + name + '_OD.npy')

                max_val = max(FD)
                print(max_val)
                FD_auc += FD.sum() / (1000*max_val)
                fa.append(FD.sum() / (1000*max_val))
                FDun_auc += FD_un.sum() / (1000*max_val)
                fua.append(FD_un.sum() / (1000*max_val))
                degree_auc += degree.sum() / (1000*max_val)
                da.append(degree.sum() / (1000*max_val))
                MS_auc += MS.sum() / (1000*max_val)
                ma.append(MS.sum() / (1000*max_val))
                adpDegree_auc += adpDegree.sum() / (1000*max_val)
                aa.append(adpDegree.sum() / (1000*max_val))
                PR_auc += PR.sum() / (1000*max_val)
                pa.append(PR.sum() / (1000*max_val))
                DND_auc += DND.sum() / (1000*max_val)
                dnda.append(DND.sum() / (1000*max_val))
                Core_auc += CoreHD.sum() / (1000*max_val)
                corea.append(CoreHD.sum() / (1000*max_val))
                degreein_auc += degree_in.sum() / (1000*max_val)
                dia.append(degree_in.sum() / (1000*max_val))
                degreeout_auc += degree_out.sum() / (1000 * max_val)
                doa.append(degree_out.sum() / (1000 * max_val))

            std = [np.std(fa, ddof=1),np.std(fua, ddof=1), np.std(corea, ddof=1), np.std(pa, ddof=1), np.std(ma, ddof=1),
                    np.std(dnda, ddof=1),
                   np.std(aa, ddof=1), np.std(da, ddof=1), np.std(dia, ddof=1), np.std(doa, ddof=1)]
            temp = [FD_auc / num, FDun_auc / num, Core_auc / num, PR_auc / num, MS_auc / num,  DND_auc / num,
                    adpDegree_auc / num,
                    degree_auc / num, degreein_auc / num, degreeout_auc / num]
            col.bar(unit_topics, temp, yerr=std, error_kw={'elinewidth': 2, 'ecolor': '0.0', 'capsize': 4}, color=color,
                    width=0.75)
            print(std, temp)
            col.set_xticklabels(unit_topics, fontsize=10, rotation=60)
            col.set_ylim(0, 0.5)
            font1 = {
                'weight': 'normal',
                'size': 12,
            }
            col.set_title(r'$ER_{\langle d \rangle=' + D[j][:len(D[j])-2] + '}$', font1, y=1.0, x=0.5)
        fig.text(0.02, 0.25, 'ANC', va='center', fontsize=12, rotation='vertical')
        axes[0][0].text(-0.27, 1.15, 'A', transform=axes[0][0].transAxes, fontsize=14, fontweight='bold', va='top',
                        ha='left')
        axes[1][0].text(-0.27, 1.1, 'B', transform=axes[1][0].transAxes, fontsize=14, fontweight='bold', va='top',
                        ha='left')
        plt.show()


    if network_name=='SF':
        unit_topics = ["DNetKey", "FINDER", "CoreHD", "Rand", 'MinSum',  "DND", 'HDA', "HD", 'ID','OD']
        dir = "SF_ER_FINDER/SF/"
        # dir = "SF_new/"
        # file_pre = network_name  # 文件以tes_开头
        file_pre = "SF_1000_"



        lamb = [r'$SF_{\lambda=2.2}$', r'$SF_{\lambda=2.4}$', r'$SF_{\lambda=2.6}$', r'$SF_{\lambda=2.8}$']

        fig, axes = plt.subplots(2, 4, figsize=(12, 7))
        fig.subplots_adjust(top=0.9, bottom=0.12, left=0.07, right=0.97, hspace=0.45)

        D = ['2.2','2.4',"2.6","2.8"]
        color = ['#403990', "#00FF00", "#888888", "#80A6E2", "#FBDD85", "#00FFFF", "#F46F43", "#CF3D3E", "#008000", "#333333"]
        num = 30
        for j, col in enumerate(axes[0]):
            fileNames = findfile(dir, file_pre + D[j]+ "_" )
            network_names = fileNames  # 调整为参数从小到大的顺序

            FD = np.zeros(1000)  # 初始化不同方法的平均GSCC曲线
            FDun = np.zeros(1000)
            degree = np.zeros(1000)
            adpdegree = np.zeros(1000)
            learn = np.zeros(1000)
            pr = np.zeros(1000)
            dnd = np.zeros(1000)
            corehd = np.zeros(1000)
            degree_in = np.zeros(1000)
            degree_out = np.zeros(1000)

            for epoch in range(num):  # 计算平均GSCC曲线
                network_name = network_names[epoch]
                temp = np.load('FINDER_directed_result/' + network_name + '_FD.npy')
                FD += temp / (num*max(temp))
                temp = np.load('FINDER_directed_un_result/' + network_name + '_FD_un.npy')
                FDun += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_Core.npy')
                corehd += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_rand.npy')
                pr += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_degree.npy')
                degree += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_adpDegree.npy')
                adpdegree += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_MS.npy')
                learn += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_DND.npy')
                dnd += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_ID.npy')
                degree_in += temp / (num*max(temp))
                temp = np.load('final_DN_result/' + network_name + '_OD.npy')
                degree_out += temp / (num*max(temp))

            x = [_ / len(FD) for _ in range(len(FD))]
            # col.set_title(r'AvgD='+D[epoch],y=0.9)
            col.set_title(lamb[j], y=0.8, x=0.5,fontsize=14)
            col.plot(x, FD, color='#403990', lw=1.8)  # 绘制平均GSCC曲线
            col.plot(x, FDun, color="#00FF00", lw=1.8)  # 绘制平均GSCC曲线
            col.plot(x, corehd, color="#888888", lw=1.2)
            col.plot(x, pr, color="#80A6E2", lw=1.2)
            col.plot(x, learn, color="#FBDD85", lw=1.2)
            col.plot(x, dnd, color="#00FFFF", lw=1.2)
            col.plot(x, adpdegree, color="#F46F43", lw=1.2)
            col.plot(x, degree, color="#CF3D3E", lw=1.2)
            col.plot(x, degree_in, color="#008000", lw=1.2)
            col.plot(x, degree_out, color="#333333", lw=1.2)
            col.tick_params(labelsize=10)
            col.set_ylim(-0.05, 1.05)
            #col.tick_params(axis='both', labelsize=12)

        fig.text(0.5, 0.51, 'Fraction of Nodes Removed', fontsize=14, ha='center')
        fig.text(0.02, 0.75, 'LSCC', va='center', fontsize=14, rotation='vertical')
        col.legend(["DNetKey", "FINDER", "CoreHD", "Rand", 'MinSum', "DND", 'HDA', "HD", "ID", "OD"], loc=1,
                   prop={'size': 10},
                   bbox_to_anchor=(0.85, 1.17), ncol=10, borderaxespad=0, fontsize=14)

        for j, col in enumerate(axes[1]):
            FD_auc = 0
            FDun_auc = 0
            degree_auc = 0
            MS_auc = 0
            adpDegree_auc = 0
            PR_auc = 0
            DND_auc = 0
            Core_auc = 0
            degreein_auc = 0
            degreeout_auc = 0
            fa = []
            fua = []
            da = []
            ma = []
            aa = []
            pa = []
            dnda = []
            corea = []
            dia = []
            doa = []
            fileNames = findfile(dir, file_pre + D[j] + "_")

            for i in range(num):
                name = fileNames[i]  # 调整为参数从小到大的顺序
                FD = np.load('FINDER_directed_result/' + name + '_FD.npy')
                FD_un = np.load('FINDER_directed_un_result/' + name + '_FD_un.npy')
                # print(back)
                degree = np.load('final_DN_result/' + name + '_degree.npy')
                MS = np.load('final_DN_result/' + name + '_MS.npy')
                adpDegree = np.load('final_DN_result/' + name + '_adpDegree.npy')
                PR = np.load('final_DN_result/' + name + '_rand.npy')
                DND = np.load('final_DN_result/' + name + '_DND.npy')
                CoreHD = np.load('final_DN_result/' + name + '_Core.npy')
                degree_in = np.load('final_DN_result/' + name + '_ID.npy')
                degree_out = np.load('final_DN_result/' + name + '_OD.npy')

                max_val = max(FD)
                print(max_val)
                FD_auc += FD.sum() / (1000*max_val)
                fa.append(FD.sum() / (1000*max_val))
                FDun_auc += FD_un.sum() / (1000*max_val)
                fua.append(FD_un.sum() / (1000*max_val))
                degree_auc += degree.sum() / (1000*max_val)
                da.append(degree.sum() / (1000*max_val))
                MS_auc += MS.sum() / (1000*max_val)
                ma.append(MS.sum() / (1000*max_val))
                adpDegree_auc += adpDegree.sum() / (1000*max_val)
                aa.append(adpDegree.sum() / (1000*max_val))
                PR_auc += PR.sum() / (1000*max_val)
                pa.append(PR.sum() / (1000*max_val))
                DND_auc += DND.sum() / (1000*max_val)
                dnda.append(DND.sum() / (1000*max_val))
                Core_auc += CoreHD.sum() / (1000*max_val)
                corea.append(CoreHD.sum() / (1000*max_val))
                degreein_auc += degree_in.sum() / (1000*max_val)
                dia.append(degree_in.sum() / (1000*max_val))
                degreeout_auc += degree_out.sum() / (1000 * max_val)
                doa.append(degree_out.sum() / (1000 * max_val))

            std = [np.std(fa, ddof=1),np.std(fua, ddof=1), np.std(corea, ddof=1), np.std(pa, ddof=1), np.std(ma, ddof=1),
                    np.std(dnda, ddof=1),
                   np.std(aa, ddof=1), np.std(da, ddof=1), np.std(dia, ddof=1), np.std(doa, ddof=1)]
            temp = [FD_auc / num, FDun_auc / num, Core_auc / num, PR_auc / num, MS_auc / num,  DND_auc / num,
                    adpDegree_auc / num,
                    degree_auc / num, degreein_auc / num, degreeout_auc / num]
            col.bar(unit_topics, temp, yerr=std, error_kw={'elinewidth': 2, 'ecolor': '0.0', 'capsize': 4}, color=color,
                    width=0.75)
            print(std, temp)
            col.set_xticklabels(unit_topics, fontsize=10, rotation=60)
            col.set_ylim(0, 0.45)
            #col.tick_params(axis='both', labelsize=12)
            font1 = {
                'weight': 'normal',
                'size': 14,
            }
            col.set_title(r'$SF_{\lambda=' + D[j]+'}$', font1, y=1.0,x=0.5)
        fig.text(0.02, 0.25, 'ANC', va='center', fontsize=14, rotation='vertical')
        axes[0][0].text(-0.27, 1.15, 'A', transform=axes[0][0].transAxes, fontsize=14, fontweight='bold', va='top',
                        ha='left')
        axes[1][0].text(-0.27, 1.1, 'B', transform=axes[1][0].transAxes, fontsize=14, fontweight='bold', va='top',
                        ha='left')
        plt.show()





def bar_real():
    unit_topics = ["DNetKey", "FINDER"]
    #dir = "bar_source/"
    #file_pre = "SF_1000"

    dir = "real_network/"  # 真实数据
    file_pre = ""
    fileNames = findfile(dir, file_pre)
    fileNames = [ 'FoodWebs_little_rock', 'FoodWebs_reef',
                   'Genetic_net_m_tuberculosis', 'Genetic_net_p_aeruginosa',

                    "Friendship-network_data_2013", 'Social-leader2Inter', 'Social_net_social_prison',
                  "Trade_net_trade_basic", 'Trade_net_trade_food',
                  "subelj_cora.e",'TexasPowerGrid',  "ia-crime-moreno",'Neural_net_celegans_neural',
                  'TRN-EC-RDB64','Wiki-Vote']

    lamb = ['Food\nWebs01','Food\nWebs02',
             'Gene01','Gene02',
             'Social01','Social02',"Social03",'Trade01', 'Trade02',"Scholar",
              'Power\nGrid',  "Crime",'Neural','TRN',"Wiki-Vote", 'Average']
    print(fileNames)
    fig, axes = plt.subplots(figsize=(14, 4))   #12/7
    fig.subplots_adjust(top=0.98, bottom=0.14, left=0.055, right=0.98)
    data = np.zeros(shape=(len(fileNames)+1, 2))

    for epoch in range(len(fileNames)):
        name = fileNames[epoch]  # [:-4]
        print(name)
        #finder_un = np.load('final_DN_result/' + name + '_finder.npy')
        finder_un = np.load('FINDER_directed_un_result/' + name + '_FD_un.npy')
        #print(finder_un)
        finder_un = finder_un / finder_un[0]

        #finder = np.load('final_DN_result/' + name + '_FD.npy')   #FD0是目前最好的版本
        finder = np.load('FINDER_directed_result/' + name + '_FD.npy')
        #print(finder)
        finder = finder / finder[0]


        #finder_un = np.load('final_DN_result/' + name + '_FDun.npy')
        #print(name)
        #print(finder)

        finder_auc = round(finder.sum() / len(finder), 4)
        finder_un_auc = round(finder_un.sum() / len(finder_un), 4)

        data[epoch, :] = np.array([finder_auc, finder_un_auc])

    data[epoch + 1, :] = data.mean(axis=0)

    N = len(fileNames)
    # 创建索引位置
    index = np.arange(N+1)
    width = 0.3  # 柱子的宽度
    color = ['#403990',  "#00FF00"]
    print("directed", data[:,0])
    print("undirected", data[:, 1])
    plt.bar(index, data[:,0], width, label='DNetKey',color=color[0])
    plt.bar(index + width, data[:, 1], width, label='FINDER', color=color[1])
    """
    # 在每个柱子上方标注数值
    for bar, value in zip(bars, temp):
        height = bar.get_height()
        col.text(bar.get_x() + bar.get_width() / 2, height, f'{value:.5f}', ha='center', va='bottom')
    print(std, temp)
    col.set_xticklabels(unit_topics, rotation=0)
    font1 = {
             'weight': 'normal',
             'size': 12,
             }
    col.set_title(r'$SF_{\lambda=' + D[j]+'}$', font1, y=1.0,x=0.5)
    fig.text(0.01, 0.55, 'ANC', va='center', fontsize=12, rotation='vertical')
    plt.show()
    """



    # 添加标题和标签
    plt.ylabel('ANC',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(index + width/2, lamb)  # 设置x轴标签位置
    plt.legend(bbox_to_anchor=(0.45, 0.85),prop={ 'size': 14})  # 显示图例
    # 显示图形
    plt.show()



def draw_bar_c():   #绘制多个网络的bar
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.07, right=0.97, hspace=0.4)

    dir = "ER_new_d_3-20/"
    D = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5,
         13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0]

    unit_topics = ["DNetKey", "FINDER"]
    color = ['#403990',  "#00FF00"]
    methods = {topic: {'mean': [], 'std': []} for topic in unit_topics}
    for epoch in range(len(D)):
        print("epoch",epoch)
        network_names = findfile_ER(dir, "ER_1000_", "_" + str(D[epoch]))

        finder_auc = 0
        unfinder_auc = 0
        fa = []
        unfa = []

        num = len(network_names)
        print(num)
        for i in range(len(network_names)):
            name = network_names[i]  # [:-4]
            print('网络名称：', name)

            finder = np.load('FINDER_directed_result/' + name + '_FD.npy')  # FD0是目前最好的版本
            finder_un = np.load('FINDER_directed_un_result/' + name + '_FD_un.npy')

            max_val = max(finder)
            #max_val=1
            finder_auc += finder.sum() / (1000 * max_val)
            fa.append(finder.sum() / (1000 * max_val))

            unfinder_auc += finder_un.sum() / (1000 * max_val)
            unfa.append(finder_un.sum() / (1000 * max_val))

        std = [np.std(fa, ddof=1), np.std(unfa, ddof=1)]
        temp = [finder_auc / num, unfinder_auc / num]
        for i, topic in enumerate(unit_topics):
            methods[topic]['mean'].append(temp[i])
            methods[topic]['std'].append(std[i])

    for topic, data in methods.items():
        #plt.plot(D, data['mean'],  label=topic, color=color[unit_topics.index(topic)], marker='o', linestyle='-')
        #plt.plot(D, data['mean'], label=topic, color=color[unit_topics.index(topic)], marker=None, linestyle='-')
        #plt.fill_between(D, np.array(data['mean']) - np.array(data['std']), np.array(data['mean']) + np.array(data['std']), color=color[unit_topics.index(topic)], alpha=0.2)
        axes[0].errorbar(D, data['mean'], yerr=data['std'], label=topic, marker='o',
                     color=color[unit_topics.index(topic)],  linestyle='-', elinewidth=1, capsize=5)
    axes[0].set_xlim(2.0, 21.0)
    axes[0].set_xlabel(r'$ER_{\langle d \rangle}$', fontsize=14)
    axes[0].set_ylabel('ANC', fontsize=14, labelpad=11)
    axes[0].tick_params(axis='both', labelsize=12)
    axes[0].text(0.5, 1.05, r'ER Networks', fontsize=15, ha='center', transform=axes[0].transAxes)
    axes[0].legend(bbox_to_anchor=(0.92, 0.4),fontsize=14)
    #axes[0].legend(loc='center', bbox_to_anchor=(0.5, -0.25), ncol=2, borderaxespad=0,fontsize=12)


    dir = "SF_new_lamda2-3.5/"
    D = [2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95, 3.0, 3.05, 3.1, 3.15, 3.2, 3.25, 3.3, 3.35, 3.4, 3.45, 3.5]
    file_list = []
    for i in range(len(D)):
        file_list.append("SF_1000_" + str(D[i]) + "_")
    print(file_list)
    print(len(file_list))

    unit_topics = ["DNetKey", "FINDER"]
    color = ['#403990',  "#00FF00"]
    methods = {topic: {'mean': [], 'std': []} for topic in unit_topics}
    for epoch in range(len(file_list)):
        print("epoch",epoch)
        file_pre = file_list[epoch]
        print(file_pre)
        network_names = findfile(dir, file_pre)

        finder_auc = 0
        unfinder_auc = 0
        fa = []
        unfa = []

        num = len(network_names)
        for i in range(len(network_names)):
            name = network_names[i]  # [:-4]
            print('网络名称：', name)

            finder = np.load('FINDER_directed_result/' + name + '_FD.npy')  # FD0是目前最好的版本
            finder_un = np.load('FINDER_directed_un_result/' + name + '_FD_un.npy')

            max_val = max(finder)
            #max_val=1
            finder_auc += finder.sum() / (1000 * max_val)
            fa.append(finder.sum() / (1000 * max_val))

            unfinder_auc += finder_un.sum() / (1000 * max_val)
            unfa.append(finder_un.sum() / (1000 * max_val))

        std = [np.std(fa, ddof=1), np.std(unfa, ddof=1)]
        temp = [finder_auc / num, unfinder_auc / num]
        for i, topic in enumerate(unit_topics):
            methods[topic]['mean'].append(temp[i])
            methods[topic]['std'].append(std[i])

    for topic, data in methods.items():
        #plt.plot(D, data['mean'],  label=topic, color=color[unit_topics.index(topic)], marker='o', linestyle='-')
        #plt.plot(D, data['mean'], label=topic, color=color[unit_topics.index(topic)], marker=None, linestyle='-')
        #plt.fill_between(D, np.array(data['mean']) - np.array(data['std']), np.array(data['mean']) + np.array(data['std']), color=color[unit_topics.index(topic)], alpha=0.2)
        axes[1].errorbar(D, data['mean'], yerr=data['std'], label=topic, marker='o',
                     color=color[unit_topics.index(topic)],  linestyle='-', elinewidth=1, capsize=5)
    axes[1].set_xlim(1.9, 3.6)
    axes[1].tick_params(axis='both', labelsize=12)
    axes[1].set_xlabel(r'$SF_{\lambda}$', fontsize=14)
    axes[1].set_ylabel('ANC', fontsize=14)
    axes[1].text(0.5, 1.05, r'SF Networks', fontsize=15, ha='center', transform=axes[1].transAxes)
    #axes[1].legend(bbox_to_anchor=(0.98, 1))


    axes[0].text(-0.06, 1.12, 'A', transform=axes[0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    axes[1].text(-0.06, 1.12, 'B', transform=axes[1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    plt.show()




def draw_heatmap():
    network_names = ['FoodWebs_little_rock', 'FoodWebs_reef',
                 'Genetic_net_m_tuberculosis', 'Genetic_net_p_aeruginosa',

                 "Friendship-network_data_2013", 'Social-leader2Inter', 'Social_net_social_prison',
                 "Trade_net_trade_basic", 'Trade_net_trade_food',
                 "subelj_cora.e", 'TexasPowerGrid', "ia-crime-moreno", 'Neural_net_celegans_neural',
                 'TRN-EC-RDB64', 'Wiki-Vote','Average']

    #methods = ['STLD', 'CoreHD', 'PageRk', 'MinSum', 'FINDER', 'DND', 'HDA', 'HD']
    methods = ["FD", 'FD_UN', "CoreHD", "PageRk", 'MinSum', "DND", 'HDA', "HD", 'ID', 'OD']

    lamb = ['Food\nWebs01', 'Food\nWebs02',
            'Gene01', 'Gene02',
            'Social01', 'Social02', "Social03", 'Trade01', 'Trade02', "Scholar",
            'Power\nGrid', "Crime", 'Neural', 'TRN', "Wiki-Vote", 'Average']

    # 数据矩阵
    data = np.zeros(shape=(len(methods), len(network_names)))

    # 填充数据矩阵
    for epoch in range(len(network_names) - 1):
        network_name = network_names[epoch]
        print('网络名称：', network_name)
        FD = np.load('FINDER_directed_result/' + network_name + '_FD.npy')
        FD = FD / FD[0]
        FDun = np.load('FINDER_directed_un_result/' + network_name + '_FD_un.npy')
        FDun = FDun / FDun[0]
        CoreHD = np.load('final_DN_result/' + network_name + '_Core.npy')
        CoreHD = CoreHD / CoreHD[0]
        prank = np.load('final_DN_result/' + network_name + '_PR.npy')
        prank = prank / prank[0]
        degree = np.load('final_DN_result/' + network_name + '_degree.npy')
        degree = degree / degree[0]
        adpdegree = np.load('final_DN_result/' + network_name + '_adpDegree.npy')
        adpdegree = adpdegree / adpdegree[0]
        learn = np.load('final_DN_result/' + network_name + '_MS.npy')
        learn = learn / learn[0]
        DND = np.load('final_DN_result/' + network_name + '_DND.npy')
        DND = DND / DND[0]
        degree_in = np.load('final_DN_result/' + network_name + '_ID.npy')
        degree_in = degree_in / degree_in[0]
        degree_out = np.load('final_DN_result/' + network_name + '_OD.npy')
        degree_out = degree_out / degree_out[0]

        print(len(FD))
        print(FD[0])
        """
        data[0, epoch] = round(back.sum() / len(back), 4)
        data[1, epoch] = round(CoreHD.sum() / len(back), 4)
        data[2, epoch] = round(prank.sum() / len(back), 4)
        data[3, epoch] = round(learn.sum() / len(back), 4)
        data[4, epoch] = round(finder.sum() / len(back), 4)
        data[5, epoch] = round(DND.sum() / len(back), 4)
        data[6, epoch] = round(adpdegree.sum() / len(back), 4)
        data[7, epoch] = round(degree.sum() / len(back), 4)
        """

        """back[back < 0.1] = 0
        degree[degree < 0.1] = 0
        finder[finder < 0.1] = 0
        learn[learn < 0.1] = 0
        adpdegree[adpdegree < 0.1] = 0
        prank[prank < 0.1] = 0
        DND[DND < 0.1] = 0
        CoreHD[CoreHD < 0.1] = 0"""


        data[0, epoch] = FD.sum() / len(FD)
        data[1, epoch] = FDun.sum() / len(FD)
        data[2, epoch] = CoreHD.sum() / len(FD)
        data[3, epoch] = prank.sum() / len(FD)
        data[4, epoch] = learn.sum() / len(FD)
        data[5, epoch] = DND.sum() / len(FD)
        data[6, epoch] = adpdegree.sum() / len(FD)
        data[7, epoch] = degree.sum() / len(FD)
        data[8, epoch] = degree_in.sum() / len(FD)
        data[9, epoch] = degree_out.sum() / len(FD)


    # 添加平均值行
    data[:, len(network_names) - 1] = data.mean(axis=1)

    # 绘制热
    plt.figure(figsize=(12, 7))
    ax = sns.heatmap(data, annot=True, fmt=".3f", cmap="Blues_r", xticklabels=lamb, yticklabels=methods)
    #plt.xlabel("Networks")
    #plt.ylabel("Methods")
    plt.title("ANC values of different dismantling methods in real networks", fontsize=12)
    plt.xticks(rotation=60, ha='right', fontsize=10)
    plt.yticks(rotation=0)

    """
    polygon_coords = [
        (0 , 0.05),  # 左上角
        (11, 0.05),  # 左侧凹陷上部
        (11, 0 + 1),  # 左侧凹陷下部
        (12 , 0 + 1),  # 右侧凹陷下部
        (12, 0.05),  # 右侧凹陷上部

        (15 + 0.1, 0.05 ),  # 右上角

        (15 + 0.1, 0 + 1.2),  # 右下角


        (12, 0 + 1.2),  # 右侧凹陷上部
        (12, 1 + 1.2),  # 右侧凹陷下部
        (11, 1 + 1.2),  # 左侧凹陷下部
        (11, 0 + 1.2),  # 左侧凹陷上部
        (0, 0 + 1.2)  # 左下角
    ]
    polygon = Polygon(polygon_coords, closed=True, edgecolor='#CD5C5C', facecolor='none', linewidth=4)
    ax.add_patch(polygon)
    """

    plt.tight_layout()
    plt.show()




bar('SF')   #可选'ER'或'SF'

bar_new()       #绘制0.1、1、10K节点数的SF网络和ER网络的ANC柱形图

#bar_curve('ER')     #绘制1K节点数的ER网络在不同瓦解方法的比较图
#bar_curve('SF')     #绘制1K节点数的SF网络在不同瓦解方法的比较图

draw_bar_c()       #绘制不同平均度参数的SF网络和ER网络比较图ANC曲线

bar_real()         #绘制真实网络图

#draw_heatmap()