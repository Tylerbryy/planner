"""
Real-World Business Examples for LLM-Assisted Planner

This module demonstrates practical business applications including:
- Supply chain optimization
- Project management
- Manufacturing workflow
- Customer service routing
- Resource allocation
- Marketing campaign planning
"""

import dspy
from planner import LLMAssistedPlanner, GuidanceMode, IncrementalPlanner
import json


def setup_dspy():
    """Configure DSPy with GPT-5."""
    lm = dspy.LM('openai/gpt-5', max_tokens=50000, temperature=1.0)
    dspy.settings.configure(lm=lm)
    print("Configured DSPy with GPT-5 (max_tokens=50000, temperature=1.0)")


def supply_chain_optimization():
    """
    Supply Chain: Multi-warehouse inventory restocking and distribution.

    Business Problem: Optimize inventory distribution across warehouses
    while minimizing shipping costs and maintaining stock levels.
    """
    print("\n" + "="*70)
    print("SUPPLY CHAIN OPTIMIZATION")
    print("="*70)

    initial_state = """
    Current Inventory Status:
    - Warehouse_East: Product_A (50 units, low stock - threshold 100), Product_B (200 units, adequate)
    - Warehouse_West: Product_A (300 units, excess), Product_B (80 units, low stock - threshold 150)
    - Warehouse_Central: Product_A (150 units, adequate), Product_B (120 units, low stock)
    - Factory: Product_A (500 units available), Product_B (400 units available)

    Available Trucks:
    - Truck_1 at Factory (capacity: 200 units, cost per trip: $500)
    - Truck_2 at Warehouse_West (capacity: 150 units, cost per trip: $350)
    - Truck_3 at Warehouse_Central (capacity: 250 units, cost per trip: $400)

    Shipping Routes:
    - Factory -> Warehouse_East: 2 days, $800
    - Factory -> Warehouse_West: 3 days, $1000
    - Factory -> Warehouse_Central: 1 day, $600
    - Inter-warehouse transfers: 1 day, $400
    """

    goal_state = """
    Target Inventory Levels (by end of week):
    - All warehouses above minimum stock thresholds
    - Warehouse_East: Product_A >= 100 units, Product_B >= 150 units
    - Warehouse_West: Product_B >= 150 units
    - Warehouse_Central: Product_B >= 150 units
    - Minimize total shipping costs
    - Complete within 5 days
    """

    available_actions = """
    Available Operations:
    1. load_truck(truck_id, product, quantity, location): Load product onto truck
    2. ship_truck(truck_id, from_location, to_location): Ship truck between locations
    3. unload_truck(truck_id, product, quantity, location): Unload product at destination
    4. transfer_inventory(product, quantity, from_warehouse, to_warehouse): Direct warehouse transfer
    5. request_production(product, quantity): Request additional production from factory
    6. consolidate_shipment(truck_1, truck_2, location): Combine partial loads
    """

    domain_description = """
    Supply chain logistics domain managing multi-warehouse distribution network.
    Goal is to optimize inventory levels across warehouses while minimizing costs
    and respecting time constraints and truck capacities.
    """

    domain_constraints = """
    Business Constraints:
    - Cannot exceed truck capacity limits
    - Must respect minimum stock thresholds (critical for sales)
    - Shipping times must be considered (cannot exceed 5-day window)
    - Cost optimization is important but secondary to meeting stock levels
    - Trucks can only be at one location at a time
    - Loading/unloading takes 4 hours per operation
    - Factory production lead time: 2 days
    - Prefer consolidating shipments to reduce costs
    - Peak shipping hours: avoid 8am-10am (congestion surcharge +$100)
    """

    return {
        "initial_state": initial_state,
        "goal_state": goal_state,
        "available_actions": available_actions,
        "domain_description": domain_description,
        "domain_constraints": domain_constraints
    }


def software_project_planning():
    """
    Software Development: Plan a feature release with dependencies.

    Business Problem: Schedule development tasks with team capacity
    and technical dependencies.
    """
    print("\n" + "="*70)
    print("SOFTWARE PROJECT PLANNING")
    print("="*70)

    initial_state = """
    Project Status: New feature request - User Authentication System

    Team Capacity:
    - Backend_Team: 2 developers (available 40 hrs/week each)
    - Frontend_Team: 2 developers (available 40 hrs/week each)
    - DevOps_Team: 1 engineer (available 20 hrs/week for this project)
    - QA_Team: 1 tester (available 30 hrs/week)

    Current Sprint: Week 1 of 6-week timeline
    Completed Work: None (project just started)

    Available Resources:
    - Development environments: Ready
    - Testing environments: Need setup
    - Production environment: Needs configuration
    - CI/CD pipeline: Partially configured
    """

    goal_state = """
    Deliverables (6 weeks):
    1. Complete user authentication system deployed to production
    2. Features include: Login, Registration, Password Reset, OAuth integration
    3. All features tested and passing QA
    4. Documentation completed
    5. Zero critical bugs in production
    6. Team velocity maintained for future sprints

    Success Criteria:
    - Production deployment by Week 6
    - All security audits passed
    - Performance targets met (< 200ms response time)
    - 95% test coverage
    """

    available_actions = """
    Available Tasks:
    1. design_database_schema(component, estimated_hours): Design data models
    2. implement_backend_api(feature, estimated_hours): Build REST APIs
    3. develop_frontend_ui(feature, estimated_hours): Create user interfaces
    4. setup_infrastructure(component, estimated_hours): Configure environments
    5. write_tests(component, test_type, estimated_hours): Create test suites
    6. conduct_code_review(component, reviewer): Review code quality
    7. run_qa_testing(feature, test_type): Execute test scenarios
    8. deploy_to_environment(environment, component): Deploy code
    9. write_documentation(component, doc_type): Create documentation
    10. conduct_security_audit(component): Security review
    11. optimize_performance(component): Performance tuning
    12. assign_task(task, team_member, priority): Assign work
    """

    domain_description = """
    Software project management domain for planning and executing a feature release.
    Must coordinate multiple teams, respect dependencies between tasks,
    manage capacity constraints, and deliver on time.
    """

    domain_constraints = """
    Business & Technical Constraints:
    - Backend APIs must be completed before frontend can integrate
    - Infrastructure must be ready before deployments
    - Code reviews required before merging (adds 4 hours per feature)
    - QA testing requires completed features (cannot test incomplete work)
    - Security audit required before production deployment
    - Cannot exceed team capacity (risk of burnout)
    - Dependencies: OAuth requires working Login system first
    - Testing environments need 1 week setup time
    - Production deployment only allowed on Fridays
    - Each team member can only work on 2 tasks simultaneously
    - Critical path: DB Schema -> Backend API -> Frontend -> Testing -> Deployment
    - Regression testing needed after each major integration (8 hours)
    - Documentation must be completed before production release
    """

    return {
        "initial_state": initial_state,
        "goal_state": goal_state,
        "available_actions": available_actions,
        "domain_description": domain_description,
        "domain_constraints": domain_constraints
    }


def manufacturing_workflow():
    """
    Manufacturing: Production line scheduling and optimization.

    Business Problem: Schedule production orders across multiple machines
    with setup times and quality constraints.
    """
    print("\n" + "="*70)
    print("MANUFACTURING WORKFLOW OPTIMIZATION")
    print("="*70)

    initial_state = """
    Production Orders (this week):
    - Order_A: 500 units of Widget_X (due: Day 5, priority: High, profit: $10,000)
    - Order_B: 300 units of Widget_Y (due: Day 4, priority: Medium, profit: $6,000)
    - Order_C: 200 units of Widget_X (due: Day 6, priority: Low, profit: $4,000)
    - Order_D: 400 units of Widget_Z (due: Day 5, priority: High, profit: $12,000)

    Production Lines:
    - Line_1: Idle, can produce Widget_X (rate: 50/hour), Widget_Y (rate: 40/hour)
    - Line_2: Idle, can produce Widget_Y (rate: 60/hour), Widget_Z (rate: 45/hour)
    - Line_3: Idle, can produce Widget_X (rate: 55/hour), Widget_Z (rate: 50/hour)

    Current Inventory:
    - Raw_Material_A: 5000 units (for Widget_X, Widget_Y)
    - Raw_Material_B: 3000 units (for Widget_Z)
    - Raw_Material_C: 2000 units (for all widgets)

    Staff:
    - Shift_1 (Day): 8 hours, full crew (all lines operational)
    - Shift_2 (Evening): 6 hours, reduced crew (max 2 lines operational)
    - Shift_3 (Night): 0 hours (no production)

    Current Day: Day 1, Morning
    """

    goal_state = """
    Production Targets:
    - All high-priority orders completed by deadlines
    - Medium and low priority orders completed if possible
    - Maximize profit from completed orders
    - Minimize machine downtime
    - Maintain quality standards (reject rate < 2%)
    - Keep raw material inventory above safety stock (500 units each)
    - Complete before Day 6 end of business
    """

    available_actions = """
    Available Operations:
    1. setup_line(line_id, widget_type): Configure line for product (2 hours)
    2. start_production(line_id, order_id, quantity): Begin production run
    3. stop_production(line_id): Halt current production
    4. quality_check(line_id, sample_size): Inspect quality (30 min)
    5. switch_product(line_id, new_widget_type): Changeover (3 hours + cleaning)
    6. request_materials(material_type, quantity): Order materials (1 day lead time)
    7. schedule_maintenance(line_id, duration): Preventive maintenance
    8. allocate_staff(shift, line_assignments): Assign operators
    9. expedite_order(order_id): Rush production (adds 15% cost)
    10. batch_production(widget_type, quantity): Produce for inventory
    """

    domain_description = """
    Manufacturing production planning domain for managing multiple production lines,
    orders, and constraints. Must optimize schedule to maximize profit while
    meeting deadlines and quality standards.
    """

    domain_constraints = """
    Manufacturing Constraints:
    - Line setup/changeover requires downtime (2-3 hours)
    - Quality checks mandatory every 100 units produced
    - Cannot exceed raw material inventory
    - Each widget requires specific material ratios:
      * Widget_X: 5 units Material_A + 2 units Material_C
      * Widget_Y: 4 units Material_A + 3 units Material_C
      * Widget_Z: 6 units Material_B + 2 units Material_C
    - Production rates decrease 10% during evening shift (fatigue)
    - Minimum batch size: 50 units (economic production quantity)
    - Maximum continuous production: 12 hours (then mandatory maintenance)
    - Late orders incur penalty: $500/day for high priority, $200/day for medium
    - Overtime adds 50% labor cost
    - Cannot start new order if insufficient materials
    - Quality reject rate increases if line runs > 8 hours without checks
    - Material delivery only happens once per day (morning)
    """

    return {
        "initial_state": initial_state,
        "goal_state": goal_state,
        "available_actions": available_actions,
        "domain_description": domain_description,
        "domain_constraints": domain_constraints
    }


def customer_service_routing():
    """
    Customer Service: Ticket routing and escalation management.

    Business Problem: Route support tickets to appropriate teams
    while managing SLAs and agent capacity.
    """
    print("\n" + "="*70)
    print("CUSTOMER SERVICE TICKET ROUTING")
    print("="*70)

    initial_state = """
    Incoming Support Tickets (current queue):
    - Ticket_1: Billing issue, Priority: High, SLA: 2 hours, Customer: Enterprise
    - Ticket_2: Technical bug, Priority: Critical, SLA: 1 hour, Customer: Enterprise
    - Ticket_3: Feature request, Priority: Low, SLA: 48 hours, Customer: Standard
    - Ticket_4: Password reset, Priority: Medium, SLA: 4 hours, Customer: Standard
    - Ticket_5: Integration problem, Priority: High, SLA: 2 hours, Customer: Enterprise
    - Ticket_6: General inquiry, Priority: Low, SLA: 24 hours, Customer: Standard
    - Ticket_7: Service outage, Priority: Critical, SLA: 30 min, Customer: Enterprise
    - Ticket_8: Billing issue, Priority: Medium, SLA: 4 hours, Customer: Standard

    Support Team Status:
    - Tier_1_Agents: 3 agents available (handles: Password, General inquiry, basic Billing)
    - Tier_2_Specialists: 2 agents, 1 busy (handles: Technical bugs, Integrations, complex Billing)
    - Engineering_Team: 2 engineers, both busy (handles: Critical bugs, Service outages)
    - Billing_Department: 1 specialist available (handles: All billing issues)

    Current Time: 10:00 AM
    Team Schedule:
    - Tier_1: Available until 6 PM (8 hours)
    - Tier_2: Available until 7 PM (9 hours)
    - Engineering: On-call 24/7 but currently in sprint planning (ends 11 AM)
    - Billing: Available until 5 PM (7 hours)
    """

    goal_state = """
    Service Objectives:
    - All SLAs met (zero breaches)
    - Critical tickets resolved first
    - Enterprise customers prioritized
    - Efficient resource utilization (minimize idle time)
    - Proper escalation paths followed
    - Customer satisfaction maintained (response time + resolution quality)
    - Knowledge base updated with common issues
    - Maintain team morale (avoid overload)
    """

    available_actions = """
    Available Actions:
    1. assign_ticket(ticket_id, agent, tier): Route ticket to agent
    2. escalate_ticket(ticket_id, from_tier, to_tier, reason): Escalate to higher tier
    3. auto_resolve(ticket_id, solution): Use automated solution
    4. schedule_callback(ticket_id, agent, time): Schedule follow-up
    5. request_info(ticket_id, customer): Ask for clarification
    6. merge_tickets(ticket_1, ticket_2): Combine related tickets
    7. reprioritize(ticket_id, new_priority, reason): Adjust priority
    8. add_specialist(ticket_id, specialist_type): Loop in expert
    9. create_incident(ticket_id, severity): Declare major incident
    10. notify_customer(ticket_id, message, eta): Send status update
    11. update_knowledge_base(ticket_id, solution): Document solution
    12. reallocate_resources(from_queue, to_queue): Shift agents
    """

    domain_description = """
    Customer service ticket management domain for optimizing support operations.
    Must balance SLA compliance, customer satisfaction, resource utilization,
    and proper escalation protocols.
    """

    domain_constraints = """
    Business & Operational Constraints:
    - SLA breaches result in penalties and customer dissatisfaction
    - Enterprise customers must be prioritized over Standard
    - Tier escalation rules:
      * Tier 1 can escalate to Tier 2 or Billing
      * Tier 2 can escalate to Engineering
      * Cannot skip tiers unless Critical severity
    - Agent skill matrix:
      * Tier 1: 15 min avg for simple tickets, cannot handle technical issues
      * Tier 2: 30 min avg for complex tickets, can handle most issues
      * Engineering: 45-120 min for critical bugs, limited availability
      * Billing: 20 min avg for billing issues, only handles finance
    - Cannot assign more than 3 tickets per agent simultaneously
    - Critical tickets must have immediate response (< 15 min)
    - Auto-resolution only for known issues (password resets, common FAQs)
    - Escalations require ticket notes and context (adds 5 min)
    - Engineering team won't interrupt sprint planning except for outages
    - Customer callbacks must happen same day if promised
    - Knowledge base updates required for novel solutions (adds 10 min)
    - Response time affects customer satisfaction score
    - Agent shift changes at 2 PM and 6 PM (handoff time: 15 min)
    """

    return {
        "initial_state": initial_state,
        "goal_state": goal_state,
        "available_actions": available_actions,
        "domain_description": domain_description,
        "domain_constraints": domain_constraints
    }


def marketing_campaign_launch():
    """
    Marketing: Multi-channel campaign planning and execution.

    Business Problem: Launch coordinated marketing campaign across
    multiple channels with budget and timing constraints.
    """
    print("\n" + "="*70)
    print("MARKETING CAMPAIGN LAUNCH")
    print("="*70)

    initial_state = """
    Campaign: Q4 Product Launch - New SaaS Feature

    Budget: $50,000 total
    Timeline: 8 weeks until launch date
    Current Week: Week 1 (Planning Phase)

    Team Resources:
    - Content_Team: 2 writers (20 hrs/week each)
    - Design_Team: 1 designer (30 hrs/week)
    - Social_Media_Manager: 1 person (40 hrs/week)
    - Email_Marketing_Specialist: 1 person (25 hrs/week)
    - Paid_Ads_Manager: 1 person (30 hrs/week)
    - Analytics_Team: Shared resource (10 hrs/week for this campaign)

    Channels Available:
    - Email (existing list: 50,000 subscribers)
    - LinkedIn Ads (target: B2B decision makers)
    - Google Ads (target: search intent keywords)
    - Content Marketing (blog, case studies, whitepapers)
    - Social Media (LinkedIn, Twitter/X)
    - Webinar Platform (partner integration needed)
    - PR/Media Outreach (tech publications)

    Assets Status:
    - Product demo: Not ready (Engineering delivering Week 4)
    - Brand guidelines: Complete
    - Customer testimonials: 2 available, need 5 more
    - Case studies: 0 completed
    - Landing page: Needs creation
    - Email templates: Need design
    """

    goal_state = """
    Launch Objectives (Week 8):
    - Generate 500 qualified leads
    - Achieve 10,000 landing page visits
    - Secure 200 demo requests
    - Obtain 50 sign-ups in first week
    - Maintain cost per lead under $100
    - Achieve 25% email open rate
    - Get featured in 3 industry publications
    - Webinar with 150+ attendees
    - 90+ campaign quality score across all paid channels
    - Complete campaign analytics dashboard

    Success Metrics:
    - ROI > 300% within 6 months
    - Brand awareness increase: 40%
    - Social engagement rate > 5%
    """

    available_actions = """
    Available Campaign Actions:
    1. create_content(content_type, topic, word_count): Produce content asset
    2. design_creative(asset_type, dimensions, purpose): Create visual assets
    3. setup_landing_page(template, copy, cta): Build conversion page
    4. configure_email_campaign(segment, content, send_date): Setup email blast
    5. launch_paid_ads(platform, budget, targeting, creative): Start ad campaign
    6. schedule_social_posts(platform, content, frequency, duration): Plan social content
    7. organize_webinar(topic, speakers, date, promotion_plan): Plan event
    8. conduct_pr_outreach(publication_list, pitch_angle): Media relations
    9. setup_tracking(channel, metrics, dashboard): Configure analytics
    10. ab_test(element, variant_a, variant_b, traffic_split): Run experiments
    11. collect_testimonials(customer_list, format): Gather social proof
    12. optimize_campaign(channel, metric_to_improve): Refine performance
    13. allocate_budget(channel, amount, period): Distribute spending
    14. create_partnership(partner, collaboration_type): Strategic alliances
    """

    domain_description = """
    Marketing campaign planning domain for coordinating multi-channel product launch.
    Must optimize budget allocation, content production, team capacity, and channel
    timing to maximize lead generation and conversion.
    """

    domain_constraints = """
    Marketing & Business Constraints:
    - Cannot launch ads without landing page ready
    - Email campaigns require 1 week lead time (design + review + setup)
    - Content production times:
      * Blog post: 8 hours (writing + editing)
      * Case study: 16 hours (research + writing + design)
      * Whitepaper: 40 hours (comprehensive)
      * Social post: 1 hour (copy + image)
      * Email: 6 hours (design + copy + testing)
    - Paid ads require:
      * Creative assets ready (3-5 variations for A/B testing)
      * Landing page with conversion tracking
      * Minimum $5,000 budget per platform for effective reach
      * 2-week optimization period before scaling
    - PR outreach needs:
      * Compelling story/angle
      * Product demo available
      * Executive availability for interviews
      * 4-6 week lead time for publication
    - Webinar requirements:
      * 4 weeks promotion minimum
      * Technical setup: 1 week
      * Speaker prep: 2 weeks
      * Recording/editing: 1 week post-event
    - Budget constraints:
      * Max 40% on paid ads
      * Min 20% on content creation
      * 10% reserved for testing/optimization
      * Cannot exceed total budget ($50,000)
    - Dependencies:
      * Product demo gates: webinar content, video ads, detailed content
      * Testimonials needed before case studies
      * Landing page needed before any traffic campaigns
      * Analytics setup must happen Week 1-2
    - Team capacity cannot exceed available hours
    - Content must align with brand guidelines
    - All paid campaigns need 48-hour approval process
    - A/B testing requires minimum 1,000 visitors for statistical significance
    - Social media posts need 3-day advance scheduling
    """

    return {
        "initial_state": initial_state,
        "goal_state": goal_state,
        "available_actions": available_actions,
        "domain_description": domain_description,
        "domain_constraints": domain_constraints
    }


def event_planning():
    """
    Event Management: Corporate conference planning.

    Business Problem: Plan and execute large-scale business conference
    with venues, speakers, catering, and logistics.
    """
    print("\n" + "="*70)
    print("CORPORATE EVENT PLANNING")
    print("="*70)

    initial_state = """
    Event: Annual Tech Summit 2024
    Expected Attendees: 500 people
    Event Date: 12 weeks from now
    Current Status: Week 1 (Initial Planning)

    Budget: $150,000
    Budget Breakdown (planned):
    - Venue: $40,000
    - Catering: $30,000
    - Speakers: $25,000
    - Marketing: $20,000
    - Technology/AV: $15,000
    - Misc/Contingency: $20,000

    Team:
    - Event_Manager: 1 person (full-time)
    - Marketing_Coordinator: 1 person (part-time)
    - Logistics_Coordinator: 1 person (part-time)
    - Tech_Support: 2 people (event day only)

    Current Bookings:
    - Venue: Not booked (3 options shortlisted)
    - Keynote speakers: 0 confirmed (5 in discussions)
    - Breakout speakers: 0 confirmed
    - Catering: Not booked
    - Hotel blocks: Not reserved
    - AV equipment: Not rented
    - Event platform: Not selected

    Registration:
    - System: Not setup
    - Early bird pricing: Needs definition
    - Sponsorship packages: Not created
    """

    goal_state = """
    Event Success Criteria (12 weeks):
    - 500 attendees registered and confirmed
    - Venue booked and configured (main hall + 3 breakout rooms)
    - 1 keynote speaker + 12 breakout speakers confirmed
    - Catering arranged (breakfast, lunch, coffee breaks, reception)
    - 200 hotel rooms blocked for out-of-town guests
    - Event website live with registration
    - 5 sponsors secured (target: $50,000 in sponsorships)
    - 90%+ attendee satisfaction rating
    - Stay within budget
    - Generate 100 qualified sales leads
    - Media coverage in 5+ publications
    - Post-event recordings and content published
    """

    available_actions = """
    Event Planning Actions:
    1. book_venue(venue_name, date, capacity, configuration): Reserve space
    2. invite_speaker(speaker_name, topic, fee, travel): Engage speakers
    3. arrange_catering(vendor, menu, headcount, dietary_restrictions): Order food
    4. reserve_hotel_block(hotel, room_count, dates, negotiated_rate): Secure rooms
    5. rent_av_equipment(equipment_list, duration, vendor): Book tech
    6. setup_registration(platform, pricing_tiers, payment_processing): Enable signups
    7. create_sponsorship_package(tier, benefits, price): Define sponsor levels
    8. launch_event_marketing(channels, budget, target_audience): Promote event
    9. plan_session_schedule(sessions, rooms, time_slots): Create agenda
    10. arrange_transportation(service_type, capacity, routes): Logistics
    11. order_swag(items, quantity, branding, vendor): Promotional items
    12. setup_event_app(features, sponsor_ads, agenda_integration): Mobile platform
    13. hire_staff(role, count, duration, rate): Additional personnel
    14. coordinate_rehearsal(date, participants, agenda): Practice runs
    15. create_contingency_plan(scenario, backup_solution): Risk mitigation
    """

    domain_description = """
    Event planning and management domain for executing large-scale corporate events.
    Must coordinate multiple vendors, manage budget, ensure attendee satisfaction,
    and handle complex logistics and timing constraints.
    """

    domain_constraints = """
    Event Planning Constraints:
    - Venue must be booked minimum 8 weeks before event (popular venues)
    - Keynote speakers need:
      * 6-8 weeks advance notice
      * Contract negotiation: 1-2 weeks
      * Travel/accommodation arrangements
      * Presentation prep: 2 weeks
    - Catering requirements:
      * Final headcount needed 2 weeks before event
      * Menu selection 4 weeks before
      * Dietary restrictions must be collected during registration
      * Minimum order quantities apply
    - Hotel blocks:
      * Must reserve 10 weeks before event
      * Require commitment for 80% of rooms
      * Early release date: 3 weeks before event
    - Registration system:
      * Setup time: 2 weeks
      * Early bird deadline: 8 weeks before event
      * Payment processing setup: 1 week
      * Integration with event app: 1 week
    - Marketing timeline:
      * Initial announcement: 10 weeks before
      * Early bird campaign: 8-10 weeks before
      * Regular promotion: 4-8 weeks before
      * Last-minute push: 2-4 weeks before
      * Speaker promotion: As confirmed
    - Sponsorship:
      * Packages must be created by Week 3
      * Sponsor assets needed 4 weeks before event
      * Sponsor booth setup: 1 day before event
    - Technology:
      * AV equipment booking: 4 weeks before
      * Event app launch: 3 weeks before
      * WiFi capacity planning: 2 weeks before
      * Tech rehearsal: 1 week before
    - Budget constraints:
      * Cannot exceed $150,000 total
      * Venue deposit: 50% upfront (affects cash flow)
      * Speaker deposits: Required upon signing
      * Catering: 30-day payment terms
      * Cancellation penalties apply (vendor specific)
    - Dependencies:
      * Cannot finalize agenda without confirmed speakers
      * Cannot launch registration without pricing structure
      * Cannot market event without confirmed venue
      * Cannot confirm catering without registration numbers
    - Lead times:
      * Swag production: 4-6 weeks
      * Printing (programs, badges): 2 weeks
      * Signage: 3 weeks
      * Custom branding: 4 weeks
    - Risk factors:
      * Speaker cancellations (need backups)
      * Venue issues (have backup date)
      * Low registration (need pricing strategy)
      * Weather (outdoor components)
    """

    return {
        "initial_state": initial_state,
        "goal_state": goal_state,
        "available_actions": available_actions,
        "domain_description": domain_description,
        "domain_constraints": domain_constraints
    }


def run_business_example(example_func, guidance_mode=GuidanceMode.PREDICT):
    """
    Run a business planning example.

    Args:
        example_func: Function that returns problem specification
        guidance_mode: Which LLM guidance mode to use
    """
    problem = example_func()

    print(f"\nInitializing planner with {guidance_mode.value.upper()} mode...")

    planner = LLMAssistedPlanner(
        guidance_mode=guidance_mode,
        max_iterations=100,
        max_sub_tasks=15,  # Business problems may need more sub-tasks
        enable_refinement=True
    )

    print("\nRunning planner...")
    result = planner(
        initial_state=problem["initial_state"],
        goal_state=problem["goal_state"],
        available_actions=problem["available_actions"],
        domain_description=problem["domain_description"],
        domain_constraints=problem["domain_constraints"]
    )

    # Display results
    print(f"\n{'='*70}")
    print("PLANNING RESULTS")
    print(f"{'='*70}")

    print(f"\n📊 Problem Decomposition:")
    print(f"   Total sub-tasks identified: {len(result.sub_tasks)}")
    for i, task in enumerate(result.sub_tasks, 1):
        print(f"   {i}. {task}")

    print(f"\n💡 Strategic Reasoning:")
    reasoning_preview = result.decomposition_reasoning[:500] + "..." if len(result.decomposition_reasoning) > 500 else result.decomposition_reasoning
    print(f"   {reasoning_preview}")

    print(f"\n📋 Execution Plan ({len(result.plan)} steps):")
    for i, action in enumerate(result.plan[:20], 1):  # Show first 20 steps
        print(f"   {i}. {action}")
    if len(result.plan) > 20:
        print(f"   ... and {len(result.plan) - 20} more steps")

    print(f"\n✅ Validation Results:")
    print(f"   Plan Valid: {result.is_valid}")
    print(f"   Quality Score: {result.plan_quality_score}")
    if result.validation_errors:
        print(f"   Errors Found: {result.validation_errors}")

    print(f"\n🔍 Planning Process:")
    completed_subtasks = sum(1 for t in result.planning_trace if t.get("status") != "failed")
    print(f"   Completed sub-tasks: {completed_subtasks}/{len(result.planning_trace)}")
    print(f"   Guidance mode used: {result.guidance_mode}")

    print(f"\n{'='*70}\n")

    return result


def compare_business_scenarios():
    """Compare different business scenarios."""
    print("\n" + "="*70)
    print("BUSINESS PLANNING SCENARIOS COMPARISON")
    print("="*70)

    scenarios = [
        ("Supply Chain", supply_chain_optimization),
        ("Software Project", software_project_planning),
        ("Manufacturing", manufacturing_workflow),
        ("Customer Service", customer_service_routing),
        ("Marketing Campaign", marketing_campaign_launch),
        ("Event Planning", event_planning)
    ]

    print("\nAvailable scenarios:")
    for i, (name, _) in enumerate(scenarios, 1):
        print(f"{i}. {name}")

    return scenarios


def main():
    """Main execution for business examples."""
    print("\n" + "="*70)
    print("LLM-ASSISTED PLANNER - REAL-WORLD BUSINESS EXAMPLES")
    print("="*70)

    setup_dspy()

    print("\n" + "="*70)
    print("AVAILABLE BUSINESS SCENARIOS")
    print("="*70)
    print("""
    1. Supply Chain Optimization
       - Multi-warehouse inventory distribution
       - Minimize costs while meeting stock levels

    2. Software Project Planning
       - Feature development with dependencies
       - Team capacity and sprint planning

    3. Manufacturing Workflow
       - Production line scheduling
       - Order prioritization and resource allocation

    4. Customer Service Routing
       - Support ticket management
       - SLA compliance and escalation

    5. Marketing Campaign Launch
       - Multi-channel coordination
       - Budget and timeline optimization

    6. Event Planning
       - Corporate conference logistics
       - Vendor coordination and scheduling
    """)

    print("\n" + "="*70)
    print("RUNNING EXAMPLE: Supply Chain Optimization")
    print("="*70)

    # Run supply chain example by default
    result = run_business_example(supply_chain_optimization, GuidanceMode.PREDICT)

    print("\n" + "="*70)
    print("HOW TO RUN OTHER EXAMPLES")
    print("="*70)
    print("""
    To run other business scenarios, use:

    # Software project planning
    run_business_example(software_project_planning, GuidanceMode.HYBRID)

    # Manufacturing optimization
    run_business_example(manufacturing_workflow, GuidanceMode.PREDICT)

    # Customer service routing
    run_business_example(customer_service_routing, GuidanceMode.INSPIRE)

    # Marketing campaign
    run_business_example(marketing_campaign_launch, GuidanceMode.HYBRID)

    # Event planning
    run_business_example(event_planning, GuidanceMode.PREDICT)

    # Compare all scenarios
    compare_business_scenarios()
    """)


if __name__ == "__main__":
    main()
