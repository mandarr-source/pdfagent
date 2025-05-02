from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm

# Define handoff tools for each agent
transfer_to_hotel_assistant = create_handoff_tool(
    agent_name="hotel_assistant",
    description="Transfer user to the hotel-booking assistant",
)
transfer_to_flight_assistant = create_handoff_tool(
    agent_name="flight_assistant",
    description="Transfer user to the flight-booking assistant",
)
transfer_to_restaurant_assistant = create_handoff_tool(
    agent_name="restaurant_assistant",
    description="Transfer user to the restaurant-booking assistant",
)
transfer_to_cab_assistant = create_handoff_tool(
    agent_name="cab_assistant",
    description="Transfer user to the cab-booking assistant",
)

# Define booking functions for each service
def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name}."

def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return f"Successfully booked a flight from {from_airport} to {to_airport}."

def book_restaurant(restaurant_name: str, date: str, time: str, guests: int):
    """Book a restaurant reservation
    
    Args:
        restaurant_name: Name of the restaurant
        date: Date of reservation (YYYY-MM-DD format)
        time: Time of reservation (HH:MM format)
        guests: Number of guests
    """
    return f"Successfully booked a table for {guests} at {restaurant_name} on {date} at {time}."

def book_cab(pickup: str, dropoff: str, time: str, cab_type: str = "standard"):
    """Book a cab
    
    Args:
        pickup: Pickup location
        dropoff: Dropoff location
        time: Pickup time (HH:MM format)
        cab_type: Type of cab (standard, premium, or SUV)
    """
    return f"Successfully booked a {cab_type} cab from {pickup} to {dropoff} at {time}."

# Create flight assistant
flight_assistant = create_react_agent(
    model="claude-3-5-sonnet-latest",
    tools=[book_flight, transfer_to_hotel_assistant, transfer_to_restaurant_assistant, transfer_to_cab_assistant],
    prompt="You are a flight booking assistant. You help users book flights between airports. If the user asks about booking hotels, restaurants, or cabs, transfer them to the appropriate assistant.",
    name="flight_assistant",
)

# Create hotel assistant
hotel_assistant = create_react_agent(
    model="claude-3-7-sonnet-latest",
    tools=[book_hotel, transfer_to_flight_assistant, transfer_to_restaurant_assistant, transfer_to_cab_assistant],
    prompt="You are a hotel booking assistant. You help users book hotel stays. If the user asks about booking flights, restaurants, or cabs, transfer them to the appropriate assistant.",
    name="hotel_assistant",
)

# Create restaurant assistant
restaurant_assistant = create_react_agent(
    model="claude-3-7-sonnet-latest",
    tools=[book_restaurant, transfer_to_flight_assistant, transfer_to_hotel_assistant, transfer_to_cab_assistant],
    prompt="You are a restaurant booking assistant. You help users book tables at restaurants. Ask for restaurant name, date, time, and number of guests if not provided. If the user asks about booking flights, hotels, or cabs, transfer them to the appropriate assistant.",
    name="restaurant_assistant",
)

# Create cab assistant
cab_assistant = create_react_agent(
    model="claude-3-5-sonnet-latest",
    tools=[book_cab, transfer_to_flight_assistant, transfer_to_hotel_assistant, transfer_to_restaurant_assistant],
    prompt="You are a cab booking assistant. You help users book cabs for transportation. Ask for pickup location, dropoff location, and time if not provided. Suggest cab types (standard, premium, SUV) when appropriate. If the user asks about booking flights, hotels, or restaurants, transfer them to the appropriate assistant.",
    name="cab_assistant",
)

# Create the swarm with all agents
swarm = create_swarm(
    agents=[flight_assistant, hotel_assistant, restaurant_assistant, cab_assistant], 
    default_active_agent="flight_assistant"
).compile()

# Example of how to use the swarm
def run_example():
    print("Starting conversation with the travel booking swarm...\n")
    for chunk in swarm.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "I need to book a flight from DEL to BLR, a stay at Taj hotel, dinner at a nice restaurant, and a cab from the airport",
                }
            ]
        }
    ):
        print(chunk)
        print("\n")

if __name__ == "__main__":
    run_example()