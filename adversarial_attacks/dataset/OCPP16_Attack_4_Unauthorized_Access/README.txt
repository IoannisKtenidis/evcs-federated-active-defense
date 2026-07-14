Attack 4 - Unauthorized Access
======================================

This dataset contains samples from an Unauthorized access attack, in which unauthorized EV charging stations try to establish connection with the CSMS. In this attack, the attacker launches 100 fake EV Charging Stations, which all try to establish OCPP connection with the CSMS, however, they are being rejected by the CSMS.

For this attack, we provide a) CICFlowMeter statistics, b) OCPPFlowMeter statistics. For each one, the corresponding PCAP and the output CSV are included.

Attacker IP: 192.168.21.225
Flow Timeout: 120s

======================================================
/CICFlowMeter
20240622_UnauthorizedAccess_filtered_ordered.pcap
20240622_UnauthorizedAccess_filtered_ordered_labelled.csv
---------------------------------------------------
Total number of flows: 642570
Total number of malicious flows: 642461
Target IP: 192.168.21.70
Labelling method: Flows containing 192.168.21.225

======================================================
/OCPPFlowMeter
20240622_UnauthorizedAccess_filtered_ordered.pcap
20240622_UnauthorizedAccess_filtered_ordered_OcppFlows_120_labelled.csv
---------------------------------------------------
Total number of flows: 642479
Total number of malicious flows: 642461
Target IP: 192.168.21.70
Labelling method: Flows containing 192.168.21.225

