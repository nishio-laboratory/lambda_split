import numpy as np
import subprocess

edge_ip = '192.168.1.225'
cloud_ip = '192.168.1.221'

exp_list = ['1_pc', '4_pc', '8_pc', '1_npc', '4_npc', '8_npc', 'edge_only', 'cloud_only']


for exp in exp_list:
    print(f'exp: {exp}')

    subprocess.run(f'tcpdump -r log/{exp}/{exp}.pcap -w log/{exp}/{exp}_uplink.pcap src host {edge_ip}', shell=True)
    subprocess.run(f'tcpdump -r log/{exp}/{exp}.pcap -w log/{exp}/{exp}_downlink.pcap dst host {edge_ip}', shell=True)

    uplink_pcap_filesize = subprocess.run(f'ls -l log/{exp}/{exp}_uplink.pcap | awk \'{{print $5}}\'', shell=True, capture_output=True).stdout.decode('utf-8').strip()
    downlink_pcap_filesize = subprocess.run(f'ls -l log/{exp}/{exp}_downlink.pcap | awk \'{{print $5}}\'', shell=True, capture_output=True).stdout.decode('utf-8').strip()
    total_pcap_filesize = subprocess.run(f'ls -l log/{exp}/{exp}.pcap | awk \'{{print $5}}\'', shell=True, capture_output=True).stdout.decode('utf-8').strip()

    uplink_pcap_filesize = int(uplink_pcap_filesize) / 1024 / 1024
    downlink_pcap_filesize = int(downlink_pcap_filesize) / 1024 / 1024
    total_pcap_filesize = int(total_pcap_filesize) / 1024 / 1024

    print(f'uplink_pcap_filesize: {uplink_pcap_filesize:.2f} MB')
    print(f'downlink_pcap_filesize: {downlink_pcap_filesize:.2f} MB')
    print(f'total_pcap_filesize: {total_pcap_filesize:.2f} MB')

    if exp == 'edge_only':
        cloud_computational_latency = 0
        communication_latency = 0
        edge_computational_latency = np.load(f'log/{exp}/first_model_inference_time_history.npy').sum() + np.load(f'log/{exp}/second_model_inference_time_history.npy').sum() + np.load(f'log/{exp}/third_model_inference_time_history.npy').sum() + np.load(f'log/{exp}/token_sampling_time_history.npy').sum()
    elif exp == 'cloud_only':
        cloud_computational_latency = np.load(f'log/{exp}/second_model_inference_time_history.npy').sum()
        communication_latency = 24.106691 - 300 * (1 / 14.624674050622959)
        edge_computational_latency = np.load(f'log/{exp}/first_model_inference_time_history.npy').sum() + np.load(f'log/{exp}/third_model_inference_time_history.npy').sum() + np.load(f'log/{exp}/token_sampling_time_history.npy').sum()
    else:
        cloud_computational_latency = np.load(f'log/{exp}/second_model_inference_time_cloud.npy').sum()
        communication_latency = np.load(f'log/{exp}/second_model_inference_time_history.npy').sum() - cloud_computational_latency
        edge_computational_latency = np.load(f'log/{exp}/first_model_inference_time_history.npy').sum() + np.load(f'log/{exp}/third_model_inference_time_history.npy').sum() + np.load(f'log/{exp}/token_sampling_time_history.npy').sum()
    
    total_latency = cloud_computational_latency + edge_computational_latency + communication_latency

    average_throughput = 300 / total_latency    
    
    print(f'communication_latency: {communication_latency} s')
    print(f'edge_computational_latency: {edge_computational_latency} s')
    print(f'cloud_computational_latency: {cloud_computational_latency} s')
    print(f'total_latency: {total_latency} s')
    print(f'average_throughput: {average_throughput} tokens/s')
    print()


exp = 'edge_only'





exp = 'cloud_only'