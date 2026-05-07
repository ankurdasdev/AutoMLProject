public class CCM_UtliityClass {

    public class FlowInput {
        @InvocableVariable(required=true description='Semicolon separated list of symptoms from Screen Flow')
        public String symptomString;
        
        @InvocableVariable(required=true description='The ID of the Parent Work Order created in the Flow')
        public Id parentWorkOrderId;
    }
    
    public class FlowOutput {
        @InvocableVariable(description='Collection of ready-to-insert Work Order Line Items')
        public List<WorkOrderLineItem> wolisToInsert;
        
        @InvocableVariable(description='Total calculated duration in minutes')
        public Decimal totalDuration;
    }

    @InvocableMethod(label='Map Symptoms & Generate WOLIs' description='Maps symptoms to WTs, queries IDs, and calculates duration')
    public static List<FlowOutput> generateWOLIs(List<FlowInput> inputs) {
        List<FlowOutput> outputs = new List<FlowOutput>();
        
        Map<String, String> symptomToWTName = new Map<String, String>{
            'Tire Blowout / Flat Tire' => 'Emergency Highway Rescue',
            'Major Mechanical Failure (Undriveable)' => 'Flatbed Towing Service',
            'EV Battery Depleted (Stranded)' => 'EV Mobile Charging',
            'Accident / Collision' => 'Post-Accident Assessment',
            'Check Engine / Warning Light' => 'Standard Mobile Diagnostic',
            'Vehicle Won\'t Start (Dead Battery)' => 'Routine Battery Swap',
            'Cracked / Broken Windshield' => 'Windshield Replacement (Mobile)',
            'Keys Lost / Fob Unresponsive' => 'Key Fob Reprogramming',
            'Screen Frozen / Software Glitch' => 'On-Site Software Update',
            'Request Temporary Replacement' => 'White-Glove Loaner Delivery'
        };
        
        for (FlowInput input : inputs) {
            FlowOutput output = new FlowOutput();
            output.wolisToInsert = new List<WorkOrderLineItem>();
            output.totalDuration = 0;
            
            if (String.isBlank(input.symptomString)) {
                outputs.add(output);
                continue;
            }
            
            List<String> selectedSymptoms = input.symptomString.split(';');
            Set<String> targetWTNames = new Set<String>();
            
            for (String symptom : selectedSymptoms) {
                for (String key : symptomToWTName.keySet()) {
                    if (symptom.contains(key)) {
                        String wtName = symptomToWTName.get(key);
                        targetWTNames.add(wtName);
                        break; 
                    }
                }
            }

            Map<String, WorkType> wtNameToRecordMap = new Map<String, WorkType>();
            for (WorkType wt : [SELECT Id, Name, EstimatedDuration FROM WorkType WHERE Name IN :targetWTNames]) {
                wtNameToRecordMap.put(wt.Name, wt);
            }

            for (String symptom : selectedSymptoms) {
                String wtName = symptomToWTName.get(symptom.trim());
                
                if (wtName != null && wtNameToRecordMap.containsKey(wtName)) {
                    WorkType wtRecord = wtNameToRecordMap.get(wtName);

                    WorkOrderLineItem newWOLI = new WorkOrderLineItem();
                    newWOLI.WorkOrderId = input.parentWorkOrderId;
                    newWOLI.WorkTypeId = wtRecord.Id;
                    newWOLI.Subject = 'Fix: ' + symptom; 
                    newWOLI.Description = 'System generated task for reported symptom: ' + symptom;

                    output.wolisToInsert.add(newWOLI);
                    
                    if (wtRecord.EstimatedDuration != null) {
                        output.totalDuration += wtRecord.EstimatedDuration;
                    }
                }
            }
            outputs.add(output);
        }
        return outputs;
    }

}