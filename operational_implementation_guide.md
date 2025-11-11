# Operational Implementation Guide
## Incrementality Study for Specialty Pharmacy Interventions

---

## Quick Start Checklist

### Week -4 to -1: Pre-Launch
- [ ] **Data Systems**
  - [ ] Confirm risk score logging is active
  - [ ] Test intervention tracking system
  - [ ] Validate outcome capture processes
  - [ ] Set up automated data pipelines

- [ ] **Team Training**
  - [ ] FRM/RSM training session completed
  - [ ] Study protocols distributed
  - [ ] FAQ document shared
  - [ ] Support hotline established

- [ ] **Technical Setup**
  - [ ] Queue randomization algorithm deployed
  - [ ] Capacity monitoring dashboard live
  - [ ] Reporting systems configured
  - [ ] Backup processes tested

### Day 1: Launch
- [ ] Send launch communication to all teams
- [ ] Activate study tracking flags
- [ ] Begin daily monitoring
- [ ] Confirm first interventions logged correctly

---

## Field Team Protocols

### For FRMs/RSMs: Daily Workflow

#### Morning (8:00 - 9:00 AM)
1. **Check Dashboard**
   - Review your assigned patient queue
   - Note capacity for the day
   - Check for any system alerts

2. **Patient Prioritization**
   ```
   Your Queue Will Show:
   - Patient ID (de-identified)
   - Risk Score (0.0-1.0)
   - Priority Level (High/Medium/Low)
   - Recommended Contact Order
   - Days Since Flag
   ```

3. **Important**: Follow the recommended contact order even if another patient seems more urgent based on your judgment

#### During Outreach (9:00 AM - 5:00 PM)

**For Each Patient Contact:**

1. **Before Calling**
   - Log start time in system
   - Review patient history
   - Note intervention type planned

2. **During Contact**
   - Follow standard protocol
   - Document engagement level (1-5 scale)
   - Note any barriers identified

3. **After Contact**
   - Log outcome immediately:
     * Successful contact (Yes/No)
     * Intervention delivered (Type)
     * Duration (minutes)
     * Next steps scheduled
   - Move to next patient in queue

**If Unable to Reach:**
- Log attempt with timestamp
- Note reason (no answer, wrong number, voicemail)
- Follow standard retry protocol
- Move to next patient

#### End of Day (5:00 - 5:30 PM)
- Complete any pending documentation
- Review tomorrow's queue
- Submit daily summary (auto-generated)

---

## Special Protocols by Design Type

### 1. Standard Operations (Primary Design)
**No special changes** - Continue normal operations with enhanced tracking

### 2. Stepped-Wedge Territories (If Applicable)

**Schedule:**
```
Wave 1 (Month 1): Territories A, B, C
Wave 2 (Month 2): Territories D, E, F
Wave 3 (Month 3): Territories G, H, I
Wave 4 (Month 4): All remaining territories
```

**If You're in a Wave Territory:**
- Continue current protocols until your wave starts
- On transition date, switch to enhanced protocol
- Attend transition training session (mandatory)
- Report any implementation challenges immediately

### 3. Priority Queue Randomization

**How It Works:**
- System automatically randomizes daily queue order
- You see patients in the order presented
- **Do not re-order based on your judgment**
- Complete patients sequentially

**Example Queue:**
```
Today's Queue (Already Randomized):
1. Patient_7823 | Risk: 0.82 | Contact by: 10 AM
2. Patient_4521 | Risk: 0.79 | Contact by: 12 PM
3. Patient_9102 | Risk: 0.85 | Contact by: 2 PM
4. Patient_3344 | Risk: 0.77 | Contact by: 4 PM
```

### 4. Threshold-Based Assignment

**If risk score threshold = 0.70:**
- Patients ≥ 0.70: Standard intervention protocol
- Patients 0.65-0.69: Monitor only (no intervention)
- Document all patients in system regardless of intervention

---

## Data Entry Requirements

### Mandatory Fields (Must Complete Same Day)

| Field | Description | Example |
|-------|-------------|---------|
| patient_id | System-generated ID | PAT_123456 |
| contact_date | Date of attempt | 2024-03-15 |
| contact_time | Time of attempt | 14:30 |
| contact_success | Was patient reached? | Yes/No |
| intervention_type | What was delivered | Phone counseling |
| duration_minutes | Length of successful contact | 12 |
| engagement_score | Patient engagement (1-5) | 4 |
| barriers_identified | Check all that apply | Cost, Side effects |
| follow_up_scheduled | Next contact planned? | Yes/No |
| frm_id | Your ID (auto-populated) | FRM_789 |

### Optional Fields (Complete When Relevant)

| Field | Description | When to Use |
|-------|-------------|------------|
| referral_made | Referred to other services | If applicable |
| benefit_issue_resolved | Insurance/copay help | If provided |
| education_materials_sent | Resources shared | If sent |
| caregiver_involved | Family member present | If yes |
| language_barrier | Interpreter needed | If yes |

---

## Queue Management Scenarios

### Scenario 1: Technical Issues
**Situation:** System is down, can't access queue

**Action:**
1. Call help desk immediately: 1-800-XXX-XXXX
2. Document patients on backup form
3. Continue with yesterday's pending patients
4. Enter all data when system returns
5. Note "SYSTEM_DOWN" in comments

### Scenario 2: Capacity Constraints
**Situation:** Too many patients for available time

**Action:**
1. Work through queue in order shown
2. Do NOT skip to "easier" patients
3. Log capacity issue in system
4. Incomplete patients roll to next day
5. Note "CAPACITY_LIMIT" for uncontacted

### Scenario 3: Urgent Clinical Issue
**Situation:** Patient mentions urgent medical need

**Action:**
1. Follow standard escalation protocol
2. Complete intervention if safe
3. Document urgency in system
4. Flag for clinical team review
5. Continue with queue after resolution

---

## Performance Metrics (Not Used for Individual Evaluation)

### What We're Measuring:
- Contact success rate by time of day
- Intervention completion rates
- Patient engagement scores
- Time to first contact after flag

### What We're NOT Measuring:
- Individual FRM performance rankings
- Speed comparisons between FRMs
- "Success" rates by FRM

**Remember:** This study measures intervention effectiveness, not individual performance

---

## Communication Protocols

### Weekly Team Meetings
**Every Tuesday, 3:00 PM**
- Review week's activities
- Discuss challenges
- Share best practices
- Q&A with study team

### Escalation Path
1. **Technical Issues**: Help Desk (1-800-XXX-XXXX)
2. **Clinical Questions**: Clinical Lead (Dr. Smith)
3. **Study Protocol**: Study Coordinator (Jane Doe)
4. **Urgent Issues**: Manager on call

### Feedback Channels
- Daily: Use in-app feedback button
- Weekly: Team meeting discussions
- Monthly: Anonymous survey
- Anytime: Email studyteam@company.com

---

## FAQs for Field Teams

**Q: Why can't I reorder my patient queue based on my experience?**
A: The randomization is critical for measuring true intervention impact. Reordering would introduce bias.

**Q: What if I know a patient needs urgent help but they're last in queue?**
A: For true medical emergencies, follow standard escalation. For non-emergencies, maintain queue order.

**Q: How long will this study protocol last?**
A: 12 months total (6 months enrollment + 6 months follow-up). Your daily workflow impact is mainly in the enrollment period.

**Q: Will this affect my performance reviews?**
A: No. Study metrics are aggregate only. Individual performance is evaluated using standard metrics.

**Q: What if I'm sick/on PTO?**
A: Your patients are automatically redistributed. Log PTO in system as usual. No special process needed.

**Q: Can I share study details with patients?**
A: Patients should not be informed they're part of a study. Continue with standard conversation approaches.

---

## Month-by-Month Timeline

### Month 1: Launch & Stabilization
- Week 1: System activation, initial troubleshooting
- Week 2-3: Process refinement based on feedback
- Week 4: First monthly review meeting

### Month 2-3: Full Operations
- Steady state operations
- Monthly reviews and adjustments
- Mid-study survey for feedback

### Month 4-6: Continued Enrollment
- Maintain protocols
- Prepare for transition to follow-up phase
- Preliminary results review (aggregate only)

### Month 7-12: Follow-up Period
- Return to standard operations
- Continue outcome tracking
- Participate in results discussion

### Month 13: Results & Implementation
- Study results presentation
- New protocols based on findings
- Recognition ceremony for participants

---

## Tools & Resources

### System Access
- **Production System**: [URL]
- **Training Environment**: [URL]
- **Dashboard**: [URL]
- **Help Desk Portal**: [URL]

### Documentation
- Full Protocol: [SharePoint Link]
- Quick Reference Card: [PDF Download]
- Training Videos: [Learning Portal]
- FAQ Updates: [Wiki Page]

### Support Contacts
| Role | Name | Contact | Hours |
|------|------|---------|-------|
| Study Coordinator | Jane Doe | x1234 | 8 AM - 6 PM |
| Technical Support | Help Desk | x5678 | 24/7 |
| Clinical Lead | Dr. Smith | x9012 | 9 AM - 5 PM |
| Data Team | Analytics | x3456 | 8 AM - 5 PM |

---

## Recognition Program

### Monthly Recognition
- **Queue Champion**: Highest completion rate
- **Documentation Star**: Best data quality
- **Team Player**: Most helpful to peers

### Study Completion Rewards
- Certificate of participation
- Contribution to publication acknowledgment
- Team celebration event
- Professional development credit

---

## Appendix: Sample Scenarios

### Sample Day for FRM Sarah

**8:30 AM**: Login, see 12 patients in queue
**9:00 AM**: Call Patient #1 - Success (15 min)
**9:20 AM**: Call Patient #2 - No answer, voicemail
**9:25 AM**: Call Patient #3 - Success (8 min)
**9:35 AM**: Call Patient #4 - Wrong number
**9:40 AM**: Document first 4 attempts
**9:50 AM**: Break
**10:00 AM**: Continue with Patient #5...

**By 5:00 PM**:
- Attempted: 12 patients
- Successful contacts: 7
- Average duration: 12 minutes
- All documented same day

### Sample Week Schedule

**Monday**
- AM: New queue review
- PM: Standard outreach
- End: Weekly planning

**Tuesday**
- AM: Outreach continues
- PM: Team meeting (3 PM)
- End: Documentation catch-up

**Wednesday-Thursday**
- Full day: Patient outreach
- Lunch: Peer consultation
- End: Daily documentation

**Friday**
- AM: Finish week's queue
- PM: Complex case follow-ups
- End: Weekly report auto-generated

---

## Remember

✅ **DO:**
- Follow queue order exactly
- Document everything same day
- Ask questions when unsure
- Report issues immediately
- Maintain patient confidentiality

❌ **DON'T:**
- Reorder patient queues
- Skip documentation
- Discuss study with patients
- Share individual results
- Work outside system

---

**Thank you for your participation in improving patient outcomes!**

*This is a living document. Check for updates monthly.*

Version 2.0 | Last Updated: January 2024 | Next Review: February 2024