@isTest
private class CCM_UtliityClassTest {

    @testSetup
    static void setupData() {
        // 1. Create Work Types that match the keys in your symptomToWTName map
        List<WorkType> workTypes = new List<WorkType>();
        workTypes.add(new WorkType(Name = 'Emergency Highway Rescue', EstimatedDuration = 60.0, DurationType = 'Minutes'));
        workTypes.add(new WorkType(Name = 'Routine Battery Swap', EstimatedDuration = 30.0, DurationType = 'Minutes'));
        workTypes.add(new WorkType(Name = 'On-Site Software Update', EstimatedDuration = 45.0, DurationType = 'Minutes'));
        insert workTypes;

        // 2. Create a Parent Work Order
        // Note: In FSL orgs, Work Order often requires a Work Type, but for this utility test, 
        // we just need the ID to satisfy the WOLI lookup.
        WorkOrder parentWO = new WorkOrder(Subject = 'Test Emergency');
        insert parentWO;
    }

    @isTest
    static void testGenerateWOLIs_Success() {
        WorkOrder wo = [SELECT Id FROM WorkOrder LIMIT 1];
        
        // Prepare multi-symptom input string
        CCM_UtliityClass.FlowInput input = new CCM_UtliityClass.FlowInput();
        input.parentWorkOrderId = wo.Id;
        input.symptomString = 'Tire Blowout / Flat Tire; Vehicle Won\'t Start; Screen Frozen / Software Glitch';

        List<CCM_UtliityClass.FlowInput> inputs = new List<CCM_UtliityClass.FlowInput>{input};

        Test.startTest();
        List<CCM_UtliityClass.FlowOutput> outputs = CCM_UtliityClass.generateWOLIs(inputs);
        Test.stopTest();

        // Assertions
        System.assertEquals(1, outputs.size(), 'Should return exactly one output for one input');
        CCM_UtliityClass.FlowOutput result = outputs[0];

        // 60 (Tire) + 30 (Battery) + 45 (Software) = 135
        System.assertEquals(135.0, result.totalDuration, 'Total duration should be sum of the three matched Work Types');
        System.assertEquals(3, result.wolisToInsert.size(), 'Should have generated 3 WOLI records');
        
        for(WorkOrderLineItem woli : result.wolisToInsert) {
            System.assertEquals(wo.Id, woli.WorkOrderId, 'WOLI should be linked to the parent Work Order ID');
            System.assertNotEquals(null, woli.WorkTypeId, 'WOLI must have a Work Type ID assigned');
        }
    }

    @isTest
    static void testGenerateWOLIs_EmptyInput() {
        CCM_UtliityClass.FlowInput input = new CCM_UtliityClass.FlowInput();
        input.parentWorkOrderId = null; // Testing resilience
        input.symptomString = ''; 

        List<CCM_UtliityClass.FlowInput> inputs = new List<CCM_UtliityClass.FlowInput>{input};

        Test.startTest();
        List<CCM_UtliityClass.FlowOutput> outputs = CCM_UtliityClass.generateWOLIs(inputs);
        Test.stopTest();

        System.assertEquals(0, outputs[0].wolisToInsert.size(), 'Empty string should result in zero WOLIs');
        System.assertEquals(0, outputs[0].totalDuration, 'Duration should be zero');
    }

    @isTest
    static void testGenerateWOLIs_PartialMatch() {
        WorkOrder wo = [SELECT Id FROM WorkOrder LIMIT 1];
        
        // One valid symptom, one invalid symptom (not in map)
        CCM_UtliityClass.FlowInput input = new CCM_UtliityClass.FlowInput();
        input.parentWorkOrderId = wo.Id;
        input.symptomString = 'Tire Blowout / Flat Tire; Random Unmapped Symptom';

        List<CCM_UtliityClass.FlowInput> inputs = new List<CCM_UtliityClass.FlowInput>{input};

        Test.startTest();
        List<CCM_UtliityClass.FlowOutput> outputs = CCM_UtliityClass.generateWOLIs(inputs);
        Test.stopTest();

        // Only "Tire Blowout" exists in our setup data and map
        System.assertEquals(1, outputs[0].wolisToInsert.size(), 'Should only generate WOLI for the matched symptom');
        System.assertEquals(60.0, outputs[0].totalDuration, 'Duration should only reflect the matched symptom');
    }
}