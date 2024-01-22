# List Decoder for Polar Codes, CRC-Polar Codes, and PAC Codes



如果您发现此算法有用，请引用以下论文。谢谢。

刘志强，刘志强，“基于极化调整的卷积(PAC)编码:序列译码与列表译码”，载于《IEEE车载技术汇刊》，第70卷，第1期。2, pp. 1434-1447, 2021年2月，doi: 10.1109/TVT.2021.3052550。

https://ieeexplore.ieee.org/abstract/document/9328621

描述:
这是一个对polar码、CRC-polar码和PAC码的连续取消列表(SCL)解码算法的实现，可以选择各种代码构造/速率配置。
表解码算法是一种自适应的两阶段连续取消表(SCL)算法。这意味着它首先尝试L=1，然后尝试L=L_max。性能与使用L_max的列表解码相同。这个技巧已经在simulator.py文件中实现。其余文件与标准列表解码算法相同。

主文件是simulator.py，你可以在其中设置代码和通道的参数。

要在解码极性码和PAC码之间切换，您需要将极性码的生成器多项式conv_gen更改为conv_gen=[1]或任何其他多项式，如conv_gen=[1,0,1,1,0,1,1]。

If you find this algorithm useful, please cite the following paper. Thanks.

M. Rowshan, A. Burg and E. Viterbo, "Polarization-Adjusted Convolutional (PAC) Codes: Sequential Decoding vs List Decoding," in IEEE Transactions on Vehicular Technology, vol. 70, no. 2, pp. 1434-1447, Feb. 2021, doi: 10.1109/TVT.2021.3052550.

https://ieeexplore.ieee.org/abstract/document/9328621

Description:
This is an implementation of the successive cancellation list (SCL) decoding algorithm for polar codes, CRC-polar codes, and PAC codes with the choice of various code constructions/rate-profiles in Python.
The list decoding algorithm is an adaptive two stage successive cancellation list (SCL) algorithm. That means first it tries L=1 and then L=L_max. The performance is the same as list decoding with L_max. This trick has been implemented in the simulator.py file. The rest of the files are the same as the standard list decoding algorithm.

The main file is simulator.py in which you can set the parameters of the code and the channel.

To switch between decoding polar codes and PAC codes, you need to change the generator polynomial conv_gen to conv_gen=[1] for polar codes or any other polynomial such as conv_gen=[1,0,1,1,0,1,1].

## The differences between PAC codes and Polar codes are in

- 编码过程，我们还有一个阶段，我们称之为卷积预编码或预变换。如果conv_gen =[1]，则表示不进行预编码，因此模拟结果是polar码，而不是PAC码。如果你查看polar_code.py文件中的pac_encode()方法，你会发现卷积预编码方法conv_encode()， U = pcfun。conv_encode(V, conv_gen, mem)。

- 解码过程，我们对每个v同时考虑0和1值，并根据当前状态通过conv_1bit()方法获得相应的u。然后，我们基于LLR和这些u值(树上每个扩展分支的值0和1)计算路径度量。显然，我们还需要通过getNextState()方法更新当前状态。

注意，这个算法使用了“写时复制”或“延迟复制”技术。

请在ieee dot org上向mrowshan报告任何错误

- the encoding process where we have one more stage that we call convolutional precoding or pre-transformation. If conv_gen = [1], that means no precoding is performed, hence the result will be the simulation for polar codes, not PAC codes. If you look at  pac_encode() method in polar_code.py file, you would find the convolutional precoding method, conv_encode(), as U = pcfun.conv_encode(V, conv_gen, mem).
- the decoding process where we consider both 0 and 1 values for every v and obtain the corresponding u by conv_1bit() method based on the current state. Then, we calculate the path metric based on the LLR and these u values (the values 0 and 1 of each extended branch on the tree). Obviously, we need to update the current state as well by getNextState() method.

Note that the "copy on write" or "lazy copy" technique has been used in this algorithm.

Please report any bugs to mrowshan at ieee dot org
