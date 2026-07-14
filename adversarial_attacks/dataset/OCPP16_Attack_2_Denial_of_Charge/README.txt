Attack 2 - Denial of Charge via idTag manipulation
==============================================

This dataset contains samples from a Denial of Charge attack, which is a False Data Injection (FDI) attack. In this attack, the attacker performs MiTM and ARP poisoning, and then randomizes the idTag included in the OCPP RemoteStartTransaction packets transmitted from the CSMS to the EVCSs, in order to prevent the authorization of charging transactions.

For this attack, we provide a) CICFlowMeter statistics, b) OCPPFlowMeter statistics. For each one, the corresponding PCAP and the output CSV are included.

Note about CICFlowMeter: Since in the FDI there is no way to determine the modified packets in the TCP/IP layer, we have manually indicated the malicious/modified packets by enabling the CWR TCP flag on them. Therefore, the malicios flows are those where 'CWR Flag Count' > 0.

Host performing the MiTM and ARP poisoning: 192.168.21.225

======================================================
/CICFlowMeter
20230511_Denial_of_Charge_IdTag_filtered_pcaplabelled.pcap
20230511_Denial_of_Charge_IdTag_filtered_pcaplabelled_labelled.csv
---------------------------------------------------
Total number of flows: 15637
Total number of malicious flows: 1347
Target IP: 192.168.21.128
Labelling method: Flows containing OCPP messages with the "RemoteStartTransaction" keyword
Flow Timeout: 120s

======================================================
/OCPPFlowMeter
20230511_Denial_of_Charge_IdTag_filtered.pcap
20230511_Denial_of_Charge_IdTag_filtered_OcppFlows_120_labelled.csv
---------------------------------------------------
Total number of flows: 1435
Total number of malicious flows: 1354
Target IP: 192.168.21.128
Labelling methods: Flows where 'flow_total_ocpp16_starttransaction_packets' != 0 and 'flow_total_ocpp16_authorize_not_accepted_packets' != 0 and 'flow_total_ocpp16_remotestarttransaction_packets' != 0:
Flow Timeout: 120s
