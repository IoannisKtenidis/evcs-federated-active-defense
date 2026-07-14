Attack 3 - Flodding Denial of Service Heartbeat
======================================

This dataset contains samples from Denial of Service attack, that aims to overwhelm the target with Heartbeat packets. In this attack, the attacker launches 100 fake EV Charging Stations, which all establish OCPP connection with the CSMS and send Heartbeats each 250 ms. 

For this attack, we provide a) CICFlowMeter statistics, b) OCPPFlowMeter statistics. For each one, the corresponding PCAP and the output CSV are included.

Attacker IP: 192.168.21.225
Flow Timeout: 120s

======================================================
/CICFlowMeter
20240625_Flooding_Heartbeat_filtered_ordered.pcap
20240625_Flooding_Heartbeat_filtered_ordered_labelled.csv
---------------------------------------------------
Total number of flows: 9516
Total number of malicious flows: 8700
Target IP: 192.168.21.70
Labelling method: Flows containing 192.168.21.225

======================================================
/OCPPFlowMeter
20240625_Flooding_Heartbeat_filtered_ordered.pcap
20240625_Flooding_Heartbeat_filtered_ordered_OcppFlows_120_labelled.csv
---------------------------------------------------
Total number of flows: 8787
Total number of malicious flows: 8700
Target IP: 192.168.21.70
Labelling method: Flows containing 192.168.21.225
