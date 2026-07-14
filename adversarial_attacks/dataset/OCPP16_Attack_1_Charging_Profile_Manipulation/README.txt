Attack 1 - Charging Profile Manipulation
==============================================

This dataset contains samples from a Charging Profile Manipulation attack, which is a False Data Injection (FDI) attack. In this attack, the attacker performs MiTM and ARP poisoning, and then modifies the SetChargingSchedule messages transmitted from the CSMS to the EVCSs, thus manipulating the transmitted charging profiles.

For this attack, we provide a) CICFlowMeter statistics, b) OCPPFlowMeter statistics. For each one, the corresponding PCAP and the output CSV are included.

Note about CICFlowMeter: Since in the FDI there is no way to determine the modified packets in the TCP/IP layer, we have manually indicated the malicious packets (i.e., those that include OCPP SetChargingProfile messages where limit >= 90) by enabling the CWR TCP flag on them. Therefore, the malicious flows are those where 'CWR Flag Count' > 0.

Host performing the MiTM and ARP poisoning: 192.168.21.225

======================================================
/CICFlowMeter
20230502_ChargingProfile_Manipulation_filtered_pcaplabelled.pcap
20230502_ChargingProfile_Manipulation_filtered_pcaplabelled_labelled.csv
---------------------------------------------------
Total number of flows: 17660
Total number of malicious flows: 883
Target IP: 192.168.21.128
Labelling method: Flows having CWR Flag Count > 0 
Flow Timeout: 120s

======================================================
/OCPPFlowMeter
20230502_ChargingProfile_Manipulation_filtered.pcap
20230502_ChargingProfile_Manipulation_filtered_OcppFlows_120_labelled.csv
---------------------------------------------------
Total number of flows: 1610
Total number of malicious flows: 933
Target IP: 192.168.21.128
Labelling method: Flows that contain SetChargingProfile messages in which limit >= 90
Flow Timeout: 120s

