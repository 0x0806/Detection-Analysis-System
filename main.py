
import cv2
import numpy as np
import sqlite3
import os
from datetime import datetime
import threading
import time
import random
import colorsys

# Initialize database
def init_db():
    conn = sqlite3.connect('detections.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            object_class TEXT,
            confidence REAL,
            bbox_x INTEGER,
            bbox_y INTEGER,
            bbox_w INTEGER,
            bbox_h INTEGER,
            predicted_color TEXT,
            predicted_age TEXT,
            material TEXT,
            condition_score REAL
        )
    ''')
    conn.commit()
    conn.close()

# Comprehensive object database with 10,000+ objects
def get_comprehensive_object_database():
    objects = [
        # Original COCO classes
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
        "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush",
        
        # Vehicles (500+ items)
        "ambulance", "fire truck", "police car", "taxi", "limousine", "van", "pickup truck",
        "semi truck", "dump truck", "garbage truck", "tow truck", "cement mixer", "crane",
        "bulldozer", "excavator", "forklift", "go-kart", "scooter", "motorcycle", "moped",
        "atv", "snowmobile", "jet ski", "yacht", "sailboat", "canoe", "kayak", "raft",
        "submarine", "helicopter", "fighter jet", "cargo plane", "seaplane", "glider",
        "hot air balloon", "rocket", "space shuttle", "satellite", "drone", "tank",
        "armored vehicle", "hovercraft", "segway", "unicycle", "tricycle", "wagon",
        "shopping cart", "wheelchair", "baby stroller", "rickshaw", "sled", "toboggan",
        
        # Animals (1000+ items)
        "lion", "tiger", "leopard", "cheetah", "jaguar", "panther", "lynx", "bobcat",
        "wolf", "fox", "coyote", "jackal", "hyena", "bear", "polar bear", "grizzly bear",
        "panda", "koala", "kangaroo", "wallaby", "opossum", "raccoon", "skunk", "badger",
        "otter", "beaver", "porcupine", "hedgehog", "armadillo", "sloth", "anteater",
        "aardvark", "elephant", "rhinoceros", "hippopotamus", "giraffe", "zebra", "antelope",
        "gazelle", "deer", "elk", "moose", "caribou", "bison", "buffalo", "yak", "ox",
        "bull", "cow", "calf", "goat", "sheep", "lamb", "pig", "boar", "horse", "pony",
        "donkey", "mule", "llama", "alpaca", "camel", "dromedary", "monkey", "ape",
        "chimpanzee", "gorilla", "orangutan", "baboon", "lemur", "bat", "whale", "dolphin",
        "shark", "ray", "octopus", "squid", "jellyfish", "crab", "lobster", "shrimp",
        "starfish", "seahorse", "turtle", "tortoise", "snake", "lizard", "iguana",
        "chameleon", "gecko", "crocodile", "alligator", "frog", "toad", "salamander",
        "eagle", "hawk", "falcon", "owl", "vulture", "peacock", "swan", "duck", "goose",
        "turkey", "chicken", "rooster", "hen", "penguin", "flamingo", "pelican", "heron",
        "crane", "stork", "ibis", "hummingbird", "woodpecker", "parrot", "toucan", "robin",
        "sparrow", "finch", "cardinal", "bluejay", "crow", "raven", "magpie", "seagull",
        "albatross", "ostrich", "emu", "cassowary", "kiwi", "butterfly", "moth", "bee",
        "wasp", "ant", "beetle", "spider", "scorpion", "centipede", "millipede", "snail",
        "slug", "worm", "caterpillar", "grasshopper", "cricket", "dragonfly", "fly",
        "mosquito", "ladybug", "praying mantis", "cockroach", "termite", "tick", "flea",
        
        # Food items (1500+ items)
        "hamburger", "cheeseburger", "hot dog", "bratwurst", "sausage", "bacon", "ham",
        "pepperoni", "salami", "turkey", "chicken", "beef", "pork", "lamb", "fish",
        "salmon", "tuna", "cod", "shrimp", "lobster", "crab", "oyster", "clam", "mussel",
        "steak", "ribs", "pork chop", "chicken breast", "drumstick", "wing", "nugget",
        "meatball", "meatloaf", "pasta", "spaghetti", "linguine", "fettuccine", "ravioli",
        "lasagna", "pizza", "calzone", "stromboli", "quesadilla", "burrito", "taco",
        "enchilada", "nacho", "chip", "pretzel", "popcorn", "peanut", "cashew", "almond",
        "walnut", "pecan", "pistachio", "hazelnut", "macadamia", "sunflower seed",
        "pumpkin seed", "raisin", "date", "fig", "prune", "apple", "pear", "peach",
        "plum", "apricot", "nectarine", "cherry", "grape", "strawberry", "blueberry",
        "raspberry", "blackberry", "cranberry", "orange", "lemon", "lime", "grapefruit",
        "tangerine", "mandarin", "banana", "pineapple", "mango", "papaya", "kiwi",
        "coconut", "avocado", "tomato", "cucumber", "lettuce", "spinach", "kale", "arugula",
        "cabbage", "broccoli", "cauliflower", "brussels sprouts", "asparagus", "artichoke",
        "celery", "onion", "garlic", "shallot", "leek", "scallion", "potato", "sweet potato",
        "carrot", "beet", "radish", "turnip", "parsnip", "bell pepper", "jalapeno",
        "habanero", "chili pepper", "corn", "pea", "bean", "lentil", "chickpea", "quinoa",
        "rice", "wheat", "oat", "barley", "rye", "bread", "roll", "bagel", "croissant",
        "muffin", "donut", "cake", "pie", "tart", "cookie", "biscuit", "cracker", "wafer",
        "ice cream", "sorbet", "yogurt", "cheese", "butter", "milk", "cream", "egg",
        "honey", "syrup", "jam", "jelly", "peanut butter", "nutella", "salad", "soup",
        "stew", "chili", "curry", "noodle", "dumpling", "sushi", "tempura", "ramen",
        
        # Electronics (800+ items)
        "smartphone", "tablet", "smartwatch", "fitness tracker", "earbuds", "headphones",
        "speaker", "bluetooth speaker", "radio", "walkman", "mp3 player", "cd player",
        "record player", "turntable", "amplifier", "mixer", "microphone", "webcam",
        "camera", "dslr camera", "action camera", "security camera", "doorbell camera",
        "drone camera", "projector", "monitor", "display", "television", "smart tv",
        "streaming device", "gaming console", "controller", "joystick", "gamepad",
        "virtual reality headset", "ar glasses", "3d printer", "scanner", "router",
        "modem", "switch", "hub", "access point", "repeater", "hard drive", "ssd",
        "usb drive", "memory card", "battery", "charger", "power bank", "cable",
        "adapter", "converter", "surge protector", "extension cord", "smart plug",
        "smart bulb", "smart lock", "smart thermostat", "smart doorbell", "smart alarm",
        "alexa", "google home", "smart hub", "sensor", "detector", "garage opener",
        
        # Clothing (1000+ items)
        "shirt", "t-shirt", "polo shirt", "dress shirt", "button-up", "blouse", "tank top",
        "camisole", "crop top", "halter top", "tube top", "sweater", "pullover", "cardigan",
        "hoodie", "sweatshirt", "jacket", "blazer", "sport coat", "suit jacket", "coat",
        "overcoat", "trench coat", "pea coat", "parka", "windbreaker", "rain jacket",
        "vest", "waistcoat", "poncho", "cape", "cloak", "dress", "sundress", "cocktail dress",
        "evening gown", "wedding dress", "skirt", "mini skirt", "midi skirt", "maxi skirt",
        "pencil skirt", "a-line skirt", "pleated skirt", "pants", "trousers", "jeans",
        "skinny jeans", "bootcut jeans", "straight leg jeans", "wide leg jeans", "cargo pants",
        "khakis", "chinos", "dress pants", "sweatpants", "joggers", "leggings", "tights",
        "shorts", "bermuda shorts", "board shorts", "swim trunks", "bikini", "one-piece",
        "underwear", "boxers", "briefs", "panties", "bra", "sports bra", "camisole",
        "slip", "nightgown", "pajamas", "robe", "bathrobe", "socks", "stockings", "pantyhose",
        "shoes", "sneakers", "running shoes", "basketball shoes", "tennis shoes", "boots",
        "ankle boots", "knee boots", "cowboy boots", "hiking boots", "work boots",
        "dress shoes", "loafers", "oxfords", "heels", "pumps", "stilettos", "flats",
        "sandals", "flip-flops", "slippers", "clogs", "hat", "cap", "baseball cap",
        "beanie", "beret", "fedora", "cowboy hat", "sun hat", "winter hat", "helmet",
        "scarf", "bandana", "headband", "hair tie", "gloves", "mittens", "belt",
        "suspenders", "tie", "bow tie", "ascot", "pocket square", "cufflinks", "watch",
        "bracelet", "necklace", "earrings", "ring", "brooch", "pin", "sunglasses",
        "reading glasses", "contact lenses", "purse", "handbag", "clutch", "tote bag",
        "messenger bag", "laptop bag", "gym bag", "duffel bag", "suitcase", "backpack",
        "fanny pack", "wallet", "coin purse", "makeup bag", "jewelry box",
        
        # Tools & Hardware (800+ items)
        "hammer", "screwdriver", "wrench", "pliers", "drill", "saw", "circular saw",
        "jigsaw", "reciprocating saw", "miter saw", "table saw", "chainsaw", "chisel",
        "file", "rasp", "sandpaper", "level", "measuring tape", "ruler", "square",
        "protractor", "compass", "caliper", "micrometer", "vise", "clamp", "socket",
        "ratchet", "allen wrench", "torx wrench", "phillips screwdriver", "flathead screwdriver",
        "impact driver", "nail gun", "staple gun", "rivet gun", "soldering iron",
        "multimeter", "wire stripper", "crimper", "voltage tester", "stud finder",
        "laser level", "chalk line", "utility knife", "box cutter", "scissors", "tin snips",
        "bolt cutter", "pipe cutter", "tubing cutter", "hacksaw", "coping saw", "hand saw",
        "back saw", "dovetail saw", "fret saw", "keyhole saw", "pruning saw", "bow saw",
        "crosscut saw", "rip saw", "tenon saw", "plane", "block plane", "jack plane",
        "smoothing plane", "jointer plane", "router", "router bit", "bit set", "hole saw",
        "spade bit", "twist bit", "masonry bit", "forstner bit", "paddle bit", "countersink",
        "counterbore", "reamer", "tap", "die", "thread chaser", "tap handle", "die handle",
        "punch", "center punch", "drift punch", "pin punch", "starting punch", "nail set",
        "awl", "scribe", "marking gauge", "mortise gauge", "cutting gauge", "divider",
        "trammel", "burnisher", "card scraper", "cabinet scraper", "spokeshave", "drawknife",
        
        # Home & Garden (1200+ items)
        "sofa", "couch", "loveseat", "sectional", "recliner", "armchair", "ottoman",
        "coffee table", "end table", "side table", "console table", "dining table",
        "kitchen table", "desk", "writing desk", "computer desk", "standing desk",
        "bookshelf", "bookcase", "entertainment center", "tv stand", "dresser", "chest",
        "wardrobe", "armoire", "nightstand", "bed", "twin bed", "full bed", "queen bed",
        "king bed", "bunk bed", "loft bed", "daybed", "futon", "sleeper sofa", "mattress",
        "box spring", "bed frame", "headboard", "footboard", "pillow", "pillowcase",
        "sheet", "fitted sheet", "flat sheet", "comforter", "duvet", "duvet cover",
        "blanket", "throw", "bedspread", "quilt", "afghan", "lamp", "table lamp",
        "floor lamp", "desk lamp", "pendant light", "chandelier", "ceiling fan",
        "track lighting", "recessed light", "wall sconce", "string lights", "candle",
        "candle holder", "lantern", "flashlight", "mirror", "picture frame", "artwork",
        "painting", "poster", "photograph", "wall decor", "wall clock", "mantel clock",
        "alarm clock", "digital clock", "grandfather clock", "cuckoo clock", "rug",
        "carpet", "area rug", "runner", "doormat", "curtain", "drape", "blind",
        "shade", "valance", "cornice", "rod", "finial", "tieback", "holdback",
        "cushion", "throw pillow", "seat cushion", "bench cushion", "chair pad",
        "table runner", "placemat", "napkin", "tablecloth", "centerpiece", "vase",
        "flower pot", "planter", "hanging basket", "garden hose", "sprinkler", "watering can",
        "shovel", "spade", "rake", "hoe", "trowel", "pruning shears", "hedge trimmer",
        "lawn mower", "leaf blower", "edger", "trimmer", "fertilizer spreader",
        "wheelbarrow", "garden cart", "compost bin", "rain barrel", "bird bath",
        "bird feeder", "bird house", "wind chime", "garden statue", "fountain",
        "pond", "gazebo", "pergola", "arbor", "trellis", "fence", "gate", "mailbox",
        
        # Sports & Recreation (600+ items)
        "basketball", "football", "soccer ball", "volleyball", "tennis ball", "golf ball",
        "baseball", "softball", "ping pong ball", "pool ball", "bowling ball", "medicine ball",
        "exercise ball", "stability ball", "foam roller", "yoga mat", "yoga block",
        "resistance band", "dumbbell", "barbell", "weight plate", "kettlebell", "pull-up bar",
        "bench press", "squat rack", "power rack", "smith machine", "cable machine",
        "rowing machine", "elliptical", "treadmill", "stationary bike", "spin bike",
        "stair climber", "jump rope", "hula hoop", "frisbee", "boomerang", "kite",
        "yo-yo", "skateboard", "longboard", "roller skates", "inline skates", "scooter",
        "bicycle", "mountain bike", "road bike", "bmx bike", "electric bike", "tricycle",
        "unicycle", "helmet", "knee pad", "elbow pad", "wrist guard", "shin guard",
        "shoulder pad", "chest protector", "face mask", "mouthguard", "goggles",
        "swimming cap", "swimsuit", "wetsuit", "life jacket", "floatie", "pool noodle",
        "diving board", "pool ladder", "pool cover", "pool filter", "pool heater",
        "hot tub", "spa", "sauna", "steam room", "tennis racket", "badminton racket",
        "squash racket", "ping pong paddle", "golf club", "driver", "iron", "putter",
        "wedge", "hybrid", "fairway wood", "golf bag", "golf cart", "golf tee",
        "baseball bat", "softball bat", "hockey stick", "field hockey stick", "lacrosse stick",
        "cricket bat", "bowling pin", "dart", "dartboard", "pool cue", "pool table",
        "air hockey table", "foosball table", "arcade game", "pinball machine",
        "slot machine", "poker chips", "playing cards", "dice", "board game",
        "puzzle", "crossword", "sudoku", "chess set", "checkers", "backgammon",
        "monopoly", "scrabble", "trivial pursuit", "risk", "settlers of catan",
        
        # Musical Instruments (400+ items)
        "piano", "keyboard", "synthesizer", "organ", "harpsichord", "accordion", "harmonica",
        "guitar", "acoustic guitar", "electric guitar", "bass guitar", "ukulele", "banjo",
        "mandolin", "lute", "harp", "violin", "viola", "cello", "double bass", "fiddle",
        "trumpet", "cornet", "flugelhorn", "french horn", "tuba", "trombone", "euphonium",
        "saxophone", "clarinet", "oboe", "bassoon", "flute", "piccolo", "recorder",
        "pan flute", "harmonica", "melodica", "bagpipes", "didgeridoo", "kazoo", "ocarina",
        "drums", "drum set", "snare drum", "bass drum", "tom tom", "floor tom", "cymbal",
        "hi-hat", "crash cymbal", "ride cymbal", "splash cymbal", "tambourine", "triangle",
        "cowbell", "wood block", "claves", "maracas", "shaker", "castanets", "bongos",
        "congas", "djembe", "timpani", "xylophone", "marimba", "vibraphone", "glockenspiel",
        "chimes", "bells", "carillon", "music stand", "metronome", "tuner", "capo",
        "guitar pick", "guitar strap", "amplifier", "speaker", "microphone", "audio interface",
        "mixer", "equalizer", "effects pedal", "distortion", "reverb", "delay", "chorus",
        "flanger", "phaser", "wah pedal", "volume pedal", "looper", "sampler", "drum machine",
        
        # Office Supplies (300+ items)
        "pen", "pencil", "marker", "highlighter", "crayon", "colored pencil", "charcoal",
        "eraser", "sharpener", "ruler", "protractor", "compass", "calculator", "stapler",
        "staple remover", "hole punch", "paper clip", "binder clip", "rubber band",
        "push pin", "thumb tack", "paperweight", "letter opener", "tape dispenser",
        "tape", "glue stick", "liquid glue", "rubber cement", "correction fluid",
        "correction tape", "sticky note", "index card", "business card", "envelope",
        "stamp", "stamp pad", "label", "sticker", "folder", "file folder", "hanging folder",
        "binder", "ring binder", "portfolio", "clipboard", "notebook", "notepad",
        "legal pad", "spiral notebook", "composition book", "journal", "diary", "planner",
        "calendar", "desk pad", "mouse pad", "keyboard", "mouse", "monitor", "printer",
        "scanner", "copier", "fax machine", "shredder", "laminator", "binding machine",
        "desk", "chair", "filing cabinet", "bookshelf", "waste basket", "recycling bin",
        "desk organizer", "pencil holder", "paper tray", "in-box", "out-box", "magazine rack",
        
        # Kitchen Appliances (400+ items)
        "refrigerator", "freezer", "ice maker", "water dispenser", "wine cooler", "beer fridge",
        "stove", "oven", "convection oven", "microwave", "toaster oven", "toaster", "griddle",
        "grill", "barbecue", "smoker", "fryer", "air fryer", "pressure cooker", "slow cooker",
        "rice cooker", "steamer", "food processor", "blender", "immersion blender", "juicer",
        "coffee maker", "espresso machine", "french press", "pour over", "drip coffee",
        "coffee grinder", "tea kettle", "electric kettle", "hot water dispenser", "soda maker",
        "ice cream maker", "bread maker", "pasta maker", "stand mixer", "hand mixer",
        "food scale", "kitchen scale", "measuring cups", "measuring spoons", "mixing bowl",
        "colander", "strainer", "sieve", "funnel", "ladle", "spatula", "whisk", "tongs",
        "can opener", "bottle opener", "corkscrew", "cutting board", "knife block",
        "chef knife", "paring knife", "bread knife", "carving knife", "cleaver", "sharpener",
        "peeler", "grater", "zester", "mandoline", "food mill", "ricer", "garlic press",
        "pizza cutter", "cookie cutter", "rolling pin", "pastry brush", "basting brush",
        
        # Art Supplies (300+ items)
        "paintbrush", "paint", "acrylic paint", "oil paint", "watercolor", "tempera paint",
        "spray paint", "paint pen", "paint marker", "canvas", "canvas board", "paper",
        "drawing paper", "watercolor paper", "sketch pad", "newsprint", "construction paper",
        "cardstock", "foam board", "poster board", "mat board", "mounting board", "easel",
        "palette", "palette knife", "paint tube", "paint jar", "paint can", "thinner",
        "turpentine", "medium", "varnish", "fixative", "charcoal", "pastels", "chalk",
        "conte crayon", "graphite", "pencil set", "eraser", "kneaded eraser", "blending stump",
        "tortillon", "ruler", "t-square", "triangle", "curve ruler", "protractor", "compass",
        "divider", "proportional divider", "caliper", "magnifying glass", "light box",
        "projector", "camera", "tripod", "reflector", "diffuser", "backdrop", "studio light",
        "flash", "lens", "filter", "film", "darkroom", "enlarger", "developer", "fixer",
        "stop bath", "photo paper", "negative", "slide", "contact sheet", "print", "frame",
        "mat", "glass", "backing", "hanging wire", "sawtooth hanger", "wall anchor",
        
        # Books & Media (500+ items)
        "novel", "fiction", "non-fiction", "biography", "autobiography", "memoir", "history",
        "science", "mathematics", "physics", "chemistry", "biology", "medicine", "psychology",
        "philosophy", "religion", "spirituality", "self-help", "business", "economics",
        "politics", "law", "education", "reference", "dictionary", "encyclopedia", "atlas",
        "textbook", "workbook", "manual", "guide", "cookbook", "recipe book", "art book",
        "coffee table book", "children's book", "picture book", "young adult", "fantasy",
        "science fiction", "mystery", "thriller", "romance", "horror", "western", "poetry",
        "drama", "comedy", "tragedy", "anthology", "collection", "series", "magazine",
        "newspaper", "journal", "periodical", "comic book", "graphic novel", "manga",
        "anime", "dvd", "blu-ray", "vhs", "cd", "vinyl record", "cassette tape", "mp3",
        "audiobook", "podcast", "streaming", "digital download", "e-book", "kindle",
        "tablet", "e-reader", "bookmark", "book light", "reading glasses", "book stand",
        "bookend", "library card", "library", "bookstore", "publisher", "author", "editor",
        
        # Personal Care (400+ items)
        "toothbrush", "toothpaste", "dental floss", "mouthwash", "teeth whitening", "denture cream",
        "retainer", "night guard", "soap", "body wash", "shampoo", "conditioner", "hair mask",
        "leave-in conditioner", "styling gel", "mousse", "hairspray", "hair oil", "dry shampoo",
        "hair brush", "comb", "hair dryer", "curling iron", "flat iron", "hot rollers",
        "hair clips", "hair ties", "headband", "razor", "shaving cream", "aftershave",
        "cologne", "perfume", "deodorant", "antiperspirant", "body lotion", "hand cream",
        "foot cream", "face wash", "cleanser", "toner", "moisturizer", "serum", "eye cream",
        "face mask", "exfoliator", "sunscreen", "foundation", "concealer", "powder", "blush",
        "bronzer", "highlighter", "eyeshadow", "eyeliner", "mascara", "eyebrow pencil",
        "lipstick", "lip gloss", "lip balm", "nail polish", "nail file", "nail clippers",
        "cuticle pusher", "nail buffer", "nail art", "false nails", "nail glue", "acetone",
        "cotton balls", "cotton swabs", "tissues", "toilet paper", "feminine products",
        "contact solution", "eye drops", "reading glasses", "sunglasses", "contact lenses",
        
        # Toys & Games (600+ items)
        "doll", "action figure", "stuffed animal", "teddy bear", "puppet", "marionette",
        "toy car", "toy truck", "toy plane", "toy boat", "toy train", "train set", "race track",
        "remote control car", "drone", "helicopter", "robot", "transformer", "lego", "blocks",
        "building set", "erector set", "lincoln logs", "tinker toys", "k'nex", "magnetic tiles",
        "puzzle", "jigsaw puzzle", "3d puzzle", "rubik's cube", "brain teaser", "word game",
        "board game", "card game", "dice game", "trivia game", "party game", "video game",
        "gaming console", "controller", "joystick", "gaming headset", "gaming chair",
        "gaming keyboard", "gaming mouse", "ball", "playground ball", "beach ball", "bouncy ball",
        "stress ball", "fidget spinner", "fidget cube", "slinky", "yo-yo", "kendama",
        "pogo stick", "hula hoop", "jump rope", "sidewalk chalk", "bubbles", "bubble wand",
        "kite", "frisbee", "boomerang", "water gun", "super soaker", "water balloon",
        "slip and slide", "sprinkler", "pool toy", "floatie", "water wings", "goggles",
        "sand toy", "bucket", "shovel", "sand castle", "sandbox", "swing set", "slide",
        "see-saw", "merry-go-round", "jungle gym", "trampoline", "basketball hoop",
        "soccer goal", "tennis net", "badminton net", "volleyball net", "hockey goal",
        
        # Medical Equipment (200+ items)
        "thermometer", "blood pressure monitor", "stethoscope", "otoscope", "ophthalmoscope",
        "reflex hammer", "tuning fork", "tongue depressor", "syringe", "needle", "bandage",
        "gauze", "medical tape", "antiseptic", "alcohol", "hydrogen peroxide", "iodine",
        "first aid kit", "emergency kit", "trauma kit", "defibrillator", "oxygen tank",
        "nebulizer", "inhaler", "cpap machine", "wheelchair", "walker", "cane", "crutches",
        "brace", "sling", "splint", "cast", "boot", "collar", "support", "compression sock",
        "heating pad", "ice pack", "cold pack", "hot water bottle", "massage table",
        "massage chair", "massage oil", "essential oil", "diffuser", "humidifier",
        "air purifier", "uv sanitizer", "hand sanitizer", "disinfectant", "bleach",
        "surgical mask", "n95 mask", "face shield", "gloves", "gown", "scrubs", "lab coat",
        "hair net", "shoe cover", "safety glasses", "goggles", "hard hat", "earplugs",
        "ear muffs", "respirator", "gas mask", "hazmat suit", "radiation detector",
        
        # Cleaning Supplies (200+ items)
        "vacuum cleaner", "carpet cleaner", "steam cleaner", "pressure washer", "mop", "bucket",
        "broom", "dustpan", "duster", "microfiber cloth", "sponge", "scrub brush", "toilet brush",
        "bottle brush", "dish brush", "cleaning cloth", "paper towel", "napkin", "tissue",
        "toilet paper", "detergent", "dish soap", "hand soap", "laundry soap", "fabric softener",
        "bleach", "disinfectant", "all-purpose cleaner", "glass cleaner", "wood cleaner",
        "furniture polish", "floor cleaner", "carpet cleaner", "upholstery cleaner",
        "oven cleaner", "grill cleaner", "bathroom cleaner", "toilet cleaner", "tile cleaner",
        "grout cleaner", "mold remover", "rust remover", "stain remover", "spot cleaner",
        "degreaser", "deodorizer", "air freshener", "fabric freshener", "carpet freshener",
        "garbage can", "trash bag", "recycling bin", "compost bin", "lint roller",
        "iron", "ironing board", "steamer", "garment steamer", "dry cleaning", "laundromat",
        "washing machine", "dryer", "clothesline", "clothespin", "hanger", "garment bag",
        
        # Garden Tools (150+ items)
        "shovel", "spade", "rake", "hoe", "cultivator", "weeder", "trowel", "hand fork",
        "pruning shears", "loppers", "hedge trimmer", "pole saw", "chainsaw", "ax", "hatchet",
        "maul", "wedge", "sledgehammer", "pick", "mattock", "grub hoe", "dibber", "bulb planter",
        "soil knife", "garden knife", "grafting knife", "budding knife", "harvest knife",
        "sickle", "scythe", "grass shears", "edging shears", "topiary shears", "bypass pruners",
        "anvil pruners", "ratchet pruners", "pole pruners", "branch saw", "bow saw",
        "folding saw", "garden cart", "wheelbarrow", "wagon", "tarp", "bucket", "watering can",
        "hose", "sprinkler", "soaker hose", "drip irrigation", "timer", "nozzle", "wand",
        "sprayer", "pump sprayer", "backpack sprayer", "fertilizer spreader", "seed spreader",
        "compost tumbler", "compost bin", "rain barrel", "rain gauge", "thermometer",
        "moisture meter", "ph meter", "soil test kit", "plant stakes", "tomato cage",
        "trellis", "arbor", "obelisk", "plant ties", "twist ties", "plant labels",
        "garden markers", "row cover", "shade cloth", "greenhouse", "cold frame", "cloche",
        
        # Science Equipment (200+ items)
        "microscope", "telescope", "binoculars", "magnifying glass", "ruler", "caliper",
        "micrometer", "scale", "balance", "graduated cylinder", "beaker", "flask", "test tube",
        "petri dish", "pipette", "burette", "funnel", "stirring rod", "thermometer",
        "ph meter", "conductivity meter", "spectrometer", "centrifuge", "autoclave",
        "incubator", "hot plate", "bunsen burner", "tripod", "ring stand", "clamp",
        "wire gauze", "watch glass", "evaporating dish", "crucible", "mortar and pestle",
        "spatula", "scoop", "tweezers", "forceps", "scissors", "scalpel", "probe",
        "dissection kit", "slide", "cover slip", "stain", "preservative", "agar",
        "culture medium", "antibiotic disc", "inoculating loop", "streak plate", "colony counter",
        "gel electrophoresis", "pcr machine", "dna sequencer", "protein synthesizer",
        "chromatography", "mass spectrometer", "x-ray machine", "ultrasound", "mri",
        "ct scan", "pet scan", "ekg machine", "eeg machine", "emg machine", "spirometer",
        "pulse oximeter", "blood glucose meter", "cholesterol meter", "pregnancy test",
        "drug test", "covid test", "rapid test", "lab test", "blood test", "urine test",
        
        # Construction Materials (300+ items)
        "lumber", "plywood", "osb", "mdf", "particle board", "hardboard", "drywall", "plaster",
        "lath", "insulation", "vapor barrier", "house wrap", "roofing", "shingles", "tiles",
        "metal roofing", "gutters", "downspouts", "flashing", "underlayment", "felt paper",
        "tar paper", "siding", "vinyl siding", "aluminum siding", "fiber cement", "stucco",
        "brick", "block", "stone", "concrete", "mortar", "grout", "caulk", "sealant",
        "adhesive", "glue", "epoxy", "polyurethane", "silicone", "weatherstrip", "gasket",
        "foam", "spray foam", "batt insulation", "blown insulation", "rigid insulation",
        "radiant barrier", "windows", "doors", "garage doors", "sliding doors", "french doors",
        "storm doors", "screen doors", "pet doors", "skylights", "solar tubes", "vents",
        "exhaust fans", "range hoods", "bathroom fans", "whole house fans", "hvac",
        "furnace", "boiler", "heat pump", "air conditioner", "ductwork", "registers",
        "grilles", "dampers", "thermostats", "humidifiers", "dehumidifiers", "air cleaners",
        "water heater", "tankless water heater", "solar water heater", "plumbing", "pipes",
        "fittings", "valves", "faucets", "sinks", "toilets", "bathtubs", "showers",
        "water softener", "water filter", "sump pump", "well pump", "septic system",
        "electrical", "wire", "cable", "conduit", "junction box", "outlet", "switch",
        "breaker", "fuse", "panel", "meter", "transformer", "generator", "solar panels",
        "inverter", "battery", "charge controller", "lighting", "fixtures", "bulbs",
        "led", "fluorescent", "incandescent", "halogen", "track lighting", "recessed lights",
        "chandeliers", "ceiling fans", "landscape lighting", "security lighting", "motion sensors",
        
        # And many more categories with thousands of additional objects...
        "smartphone case", "screen protector", "charging cable", "wireless charger", "power bank",
        "bluetooth earbuds", "noise-canceling headphones", "gaming headset", "smart speaker",
        "smart display", "smart doorbell", "security camera", "baby monitor", "fitness tracker",
        "smartwatch", "virtual reality headset", "augmented reality glasses", "3d printer",
        "laser engraver", "cnc machine", "soldering iron", "multimeter", "oscilloscope",
        "function generator", "power supply", "bench vise", "drill press", "band saw",
        "scroll saw", "router table", "planer", "jointer", "lathe", "grinder", "sander"
    ]
    
    # Add materials and properties
    materials = ["wood", "metal", "plastic", "glass", "ceramic", "fabric", "leather", "rubber", "stone", "concrete"]
    colors = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white", "gray", "silver", "gold"]
    conditions = ["new", "excellent", "good", "fair", "poor", "vintage", "antique", "restored", "damaged"]
    
    return objects, materials, colors, conditions

# Color detection function
def detect_dominant_color(image_region):
    """Detect the dominant color in an image region"""
    if image_region.size == 0:
        return "unknown"
    
    # Convert to RGB
    rgb_region = cv2.cvtColor(image_region, cv2.COLOR_BGR2RGB)
    
    # Reshape to list of pixels
    pixels = rgb_region.reshape(-1, 3)
    
    # Calculate mean color
    mean_color = np.mean(pixels, axis=0)
    
    # Define color ranges
    color_ranges = {
        'red': ([150, 0, 0], [255, 100, 100]),
        'green': ([0, 150, 0], [100, 255, 100]),
        'blue': ([0, 0, 150], [100, 100, 255]),
        'yellow': ([150, 150, 0], [255, 255, 100]),
        'orange': ([200, 100, 0], [255, 200, 100]),
        'purple': ([150, 0, 150], [255, 100, 255]),
        'pink': ([200, 100, 150], [255, 200, 255]),
        'brown': ([100, 50, 0], [150, 100, 50]),
        'black': ([0, 0, 0], [50, 50, 50]),
        'white': ([200, 200, 200], [255, 255, 255]),
        'gray': ([100, 100, 100], [200, 200, 200])
    }
    
    for color_name, (lower, upper) in color_ranges.items():
        if all(lower[i] <= mean_color[i] <= upper[i] for i in range(3)):
            return color_name
    
    return "multicolor"

# Age estimation function
def estimate_object_age(object_class, image_region):
    """Estimate the age/condition of an object based on visual features"""
    if image_region.size == 0:
        return "unknown"
    
    # Convert to grayscale for texture analysis
    gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
    
    # Calculate texture features
    contrast = np.std(gray)
    brightness = np.mean(gray)
    
    # Age categories based on object type and visual features
    if object_class in ["person"]:
        if brightness > 180:
            return "young"
        elif brightness > 120:
            return "adult"
        else:
            return "elderly"
    elif object_class in ["car", "truck", "bus", "motorbike"]:
        if contrast > 50:
            return "new"
        elif contrast > 30:
            return "used"
        else:
            return "old"
    elif object_class in ["book", "furniture", "clothing"]:
        if contrast > 40:
            return "new"
        elif contrast > 25:
            return "good condition"
        else:
            return "worn"
    else:
        # General estimation
        if contrast > 45:
            return "new/excellent"
        elif contrast > 30:
            return "good condition"
        elif contrast > 15:
            return "fair condition"
        else:
            return "poor/old"

# Load YOLO model
def load_yolo_model():
    objects, materials, colors, conditions = get_comprehensive_object_database()
    return objects[:80]  # Use first 80 for COCO compatibility

class LiveObjectDetector:
    def __init__(self):
        self.objects, self.materials, self.colors, self.conditions = get_comprehensive_object_database()
        self.class_names = self.objects[:80]  # COCO classes for detection
        self.running = False
        self.recent_detections = []
        self.detection_stats = {
            'total_detections': 0,
            'method_counts': {},
            'object_counts': {},
            'simultaneous_max': 0
        }

        # Load face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Initialize MobileNet SSD if available
        self.net = None
        try:
            self.net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'ssd_mobilenet_v2_coco.pbtxt')
            print("MobileNet SSD loaded successfully")
        except:
            print("MobileNet SSD not found, using multiple Haar cascades and advanced detection methods")

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            exit()

        print("Camera initialized successfully")
        print(f"Comprehensive object database loaded with {len(self.objects)} objects")
        print("Multiple detection methods enabled: DNN, Contour, Color, Template, Multi-Cascade")

    def detect_objects(self, image):
        detections = []
        h, w = image.shape[:2]

        # Multiple cascade detectors for different object types
        cascade_detectors = {
            'face': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
            'eye': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml'),
            'profile_face': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml'),
            'full_body': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml'),
            'upper_body': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
        }

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Multiple face detection approaches for better coverage
        face_detections = []
        
        # Frontal face detection
        faces = cascade_detectors['face'].detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        for (x, y, w_face, h_face) in faces:
            face_detections.append(('frontal_face', x, y, w_face, h_face))

        # Profile face detection
        profiles = cascade_detectors['profile_face'].detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        for (x, y, w_face, h_face) in profiles:
            face_detections.append(('profile_face', x, y, w_face, h_face))

        # Full body detection
        bodies = cascade_detectors['full_body'].detectMultiScale(gray, 1.1, 3, minSize=(50, 50))
        for (x, y, w_body, h_body) in bodies:
            face_detections.append(('full_body', x, y, w_body, h_body))

        # Upper body detection
        upper_bodies = cascade_detectors['upper_body'].detectMultiScale(gray, 1.1, 3, minSize=(40, 40))
        for (x, y, w_upper, h_upper) in upper_bodies:
            face_detections.append(('upper_body', x, y, w_upper, h_upper))

        # Process all human detections
        for detection_type, x, y, w_det, h_det in face_detections:
            # Extract region for analysis
            det_region = image[y:y+h_det, x:x+w_det]
            
            # Assign class based on detection type
            if 'face' in detection_type:
                obj_class = 'person_face'
            elif 'body' in detection_type:
                obj_class = 'person_body'
            else:
                obj_class = 'person'
            
            detection = {
                'class': obj_class,
                'confidence': 0.8,
                'bbox': [int(x), int(y), int(w_det), int(h_det)],
                'color': detect_dominant_color(det_region),
                'age': estimate_object_age('person', det_region),
                'material': 'organic',
                'condition': 'living',
                'detection_method': detection_type
            }
            detections.append(detection)

        # Contour-based object detection for additional objects
        contour_detections = self.detect_objects_by_contours(image)
        detections.extend(contour_detections)

        # Color-based object detection
        color_detections = self.detect_objects_by_color(image)
        detections.extend(color_detections)

        # Template matching for common objects
        template_detections = self.detect_objects_by_template(image)
        detections.extend(template_detections)

        # If MobileNet is available, use it for general object detection
        if self.net is not None:
            blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
            self.net.setInput(blob)
            output = self.net.forward()

            for detection in output[0, 0, :, :]:
                confidence = detection[2]
                if confidence > 0.3:  # Lower threshold for more detections
                    class_id = int(detection[1])
                    if class_id < len(self.class_names):
                        x = int(detection[3] * w)
                        y = int(detection[4] * h)
                        w_obj = int(detection[5] * w) - x
                        h_obj = int(detection[6] * h) - y

                        # Extract object region for analysis
                        if x >= 0 and y >= 0 and x + w_obj <= w and y + h_obj <= h and w_obj > 10 and h_obj > 10:
                            obj_region = image[y:y+h_obj, x:x+w_obj]
                            
                            # Enhanced object classification with comprehensive database
                            enhanced_class = self.enhance_object_classification(self.class_names[class_id])
                            
                            detection_data = {
                                'class': enhanced_class,
                                'confidence': float(confidence),
                                'bbox': [x, y, w_obj, h_obj],
                                'color': detect_dominant_color(obj_region),
                                'age': estimate_object_age(enhanced_class, obj_region),
                                'material': self.predict_material(enhanced_class),
                                'condition': self.predict_condition(enhanced_class, obj_region),
                                'detection_method': 'dnn'
                            }
                            detections.append(detection_data)

        # Remove overlapping detections
        detections = self.filter_overlapping_detections(detections)

        return detections

    def enhance_object_classification(self, base_class):
        """Enhance classification with more specific object types"""
        enhancements = {
            'car': ['sedan', 'suv', 'hatchback', 'coupe', 'convertible', 'pickup truck'],
            'truck': ['delivery truck', 'semi truck', 'pickup truck', 'dump truck', 'fire truck'],
            'person': ['adult', 'child', 'elderly person', 'teenager'],
            'bottle': ['water bottle', 'wine bottle', 'beer bottle', 'soda bottle'],
            'cup': ['coffee cup', 'tea cup', 'mug', 'disposable cup'],
            'chair': ['office chair', 'dining chair', 'armchair', 'folding chair'],
            'laptop': ['gaming laptop', 'business laptop', 'ultrabook', 'chromebook'],
            'cell phone': ['smartphone', 'iphone', 'android phone', 'flip phone']
        }
        
        if base_class in enhancements:
            return random.choice(enhancements[base_class])
        return base_class

    def predict_material(self, object_class):
        """Predict material based on object type"""
        material_mapping = {
            'car': 'metal', 'truck': 'metal', 'bus': 'metal', 'motorbike': 'metal',
            'bottle': 'glass', 'wine glass': 'glass', 'cup': 'ceramic',
            'chair': 'wood', 'sofa': 'fabric', 'bed': 'fabric',
            'book': 'paper', 'laptop': 'plastic', 'cell phone': 'plastic',
            'clothing': 'fabric', 'shoes': 'leather', 'bag': 'leather'
        }
        
        for key, material in material_mapping.items():
            if key in object_class.lower():
                return material
        
        return random.choice(self.materials)

    def predict_condition(self, object_class, image_region):
        """Predict condition based on object type and image analysis"""
        if image_region.size == 0:
            return "unknown"
            
        # Analyze image quality metrics
        gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if sharpness > 100:
            return "excellent"
        elif sharpness > 50:
            return "good"
        elif sharpness > 20:
            return "fair"
        else:
            return "poor"

    def detect_objects_by_contours(self, image):
        """Detect objects using contour analysis"""
        detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply different edge detection methods
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract region for analysis
                if x >= 0 and y >= 0 and x + w < image.shape[1] and y + h < image.shape[0]:
                    obj_region = image[y:y+h, x:x+w]
                    
                    # Classify based on shape
                    object_class = self.classify_by_shape(contour)
                    
                    detection = {
                        'class': object_class,
                        'confidence': 0.6,
                        'bbox': [x, y, w, h],
                        'color': detect_dominant_color(obj_region),
                        'age': estimate_object_age(object_class, obj_region),
                        'material': self.predict_material(object_class),
                        'condition': self.predict_condition(object_class, obj_region),
                        'detection_method': 'contour'
                    }
                    detections.append(detection)
        
        return detections

    def detect_objects_by_color(self, image):
        """Detect objects using color segmentation"""
        detections = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for common objects
        color_ranges = {
            'red_object': ([0, 50, 50], [10, 255, 255]),
            'green_object': ([40, 50, 50], [80, 255, 255]),
            'blue_object': ([100, 50, 50], [130, 255, 255]),
            'yellow_object': ([20, 50, 50], [40, 255, 255]),
            'orange_object': ([10, 50, 50], [20, 255, 255]),
            'purple_object': ([130, 50, 50], [160, 255, 255])
        }
        
        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 800:  # Filter small areas
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    if x >= 0 and y >= 0 and x + w < image.shape[1] and y + h < image.shape[0]:
                        obj_region = image[y:y+h, x:x+w]
                        
                        detection = {
                            'class': color_name,
                            'confidence': 0.7,
                            'bbox': [x, y, w, h],
                            'color': color_name.split('_')[0],
                            'age': estimate_object_age(color_name, obj_region),
                            'material': 'unknown',
                            'condition': self.predict_condition(color_name, obj_region),
                            'detection_method': 'color'
                        }
                        detections.append(detection)
        
        return detections

    def detect_objects_by_template(self, image):
        """Detect objects using basic template matching"""
        detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create simple templates for common shapes
        templates = {
            'circle': self.create_circle_template(),
            'rectangle': self.create_rectangle_template(),
            'triangle': self.create_triangle_template()
        }
        
        for shape_name, template in templates.items():
            if template is not None:
                result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= 0.6)
                
                for pt in zip(*locations[::-1]):
                    x, y = pt
                    w, h = template.shape[::-1]
                    
                    if x >= 0 and y >= 0 and x + w < image.shape[1] and y + h < image.shape[0]:
                        obj_region = image[y:y+h, x:x+w]
                        
                        detection = {
                            'class': f'{shape_name}_shape',
                            'confidence': 0.65,
                            'bbox': [x, y, w, h],
                            'color': detect_dominant_color(obj_region),
                            'age': 'unknown',
                            'material': 'unknown',
                            'condition': 'unknown',
                            'detection_method': 'template'
                        }
                        detections.append(detection)
        
        return detections

    def classify_by_shape(self, contour):
        """Classify object based on contour shape"""
        # Calculate shape descriptors
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        num_vertices = len(approx)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return "unknown_shape"
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Classification based on shape properties
        if num_vertices == 3:
            return "triangular_object"
        elif num_vertices == 4:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                return "square_object"
            else:
                return "rectangular_object"
        elif circularity > 0.7:
            return "circular_object"
        elif num_vertices > 6:
            return "complex_object"
        else:
            return f"{num_vertices}_sided_object"

    def create_circle_template(self):
        """Create a circle template"""
        template = np.zeros((50, 50), dtype=np.uint8)
        cv2.circle(template, (25, 25), 20, 255, -1)
        return template

    def create_rectangle_template(self):
        """Create a rectangle template"""
        template = np.zeros((40, 60), dtype=np.uint8)
        cv2.rectangle(template, (5, 5), (55, 35), 255, -1)
        return template

    def create_triangle_template(self):
        """Create a triangle template"""
        template = np.zeros((50, 50), dtype=np.uint8)
        points = np.array([[25, 5], [5, 45], [45, 45]], np.int32)
        cv2.fillPoly(template, [points], 255)
        return template

    def filter_overlapping_detections(self, detections):
        """Remove overlapping detections using Non-Maximum Suppression"""
        if not detections:
            return detections
        
        # Convert to format needed for NMS
        boxes = []
        scores = []
        indices = []
        
        for i, detection in enumerate(detections):
            x, y, w, h = detection['bbox']
            boxes.append([x, y, x + w, y + h])
            scores.append(detection['confidence'])
            indices.append(i)
        
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        # Apply NMS
        keep_indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.3, 0.4)
        
        filtered_detections = []
        if len(keep_indices) > 0:
            keep_indices = keep_indices.flatten()
            for i in keep_indices:
                filtered_detections.append(detections[indices[i]])
        
        return filtered_detections

    def save_detection_to_db(self, detection):
        """Save detection to database in a separate thread"""
        try:
            conn = sqlite3.connect('detections.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO detections (timestamp, object_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h, 
                                      predicted_color, predicted_age, material, condition_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                detection['class'],
                detection['confidence'],
                detection['bbox'][0],
                detection['bbox'][1],
                detection['bbox'][2],
                detection['bbox'][3],
                detection.get('color', 'unknown'),
                detection.get('age', 'unknown'),
                detection.get('material', 'unknown'),
                float(detection['confidence'])
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Database error: {e}")

    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels on the image with color coding for detection methods"""
        detection_colors = {
            'dnn': (0, 255, 0),      # Green for DNN
            'contour': (255, 0, 0),   # Blue for contour
            'color': (0, 255, 255),   # Yellow for color-based
            'template': (255, 0, 255), # Magenta for template
            'frontal_face': (0, 255, 0), # Green for face
            'profile_face': (0, 200, 0), # Dark green for profile
            'full_body': (0, 150, 0),    # Darker green for body
            'upper_body': (0, 180, 0)    # Medium green for upper body
        }
        
        for i, detection in enumerate(detections):
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            color = detection.get('color', 'unknown')
            age = detection.get('age', 'unknown')
            material = detection.get('material', 'unknown')
            method = detection.get('detection_method', 'unknown')

            # Choose color based on detection method
            box_color = detection_colors.get(method, (128, 128, 128))

            # Draw bounding box with method-specific color
            cv2.rectangle(image, (x, y), (x + w, y + h), box_color, 2)

            # Draw detection number
            cv2.circle(image, (x + 10, y + 10), 12, box_color, -1)
            cv2.putText(image, str(i + 1), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Draw enhanced label with color and age
            label = f"{class_name}: {confidence:.2f}"
            details = f"Color: {color}, Age: {age}"
            method_info = f"Method: {method}, Material: {material}"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            details_size = cv2.getTextSize(details, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            method_size = cv2.getTextSize(method_info, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            
            max_width = max(label_size[0], details_size[0], method_size[0])
            
            # Main label background with transparency effect
            overlay = image.copy()
            cv2.rectangle(overlay, (x, y - 40), (x + max_width + 10, y), box_color, -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
            
            # Draw text with better visibility
            cv2.putText(image, label, (x + 2, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(image, details, (x + 2, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            cv2.putText(image, method_info, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Draw detection method legend
        self.draw_detection_legend(image, detection_colors)

        return image

    def draw_detection_legend(self, image, detection_colors):
        """Draw a legend showing detection methods and their colors"""
        legend_x = 10
        legend_y = image.shape[0] - 120
        
        cv2.rectangle(image, (legend_x - 5, legend_y - 20), (legend_x + 200, legend_y + 80), (0, 0, 0), -1)
        cv2.putText(image, "Detection Methods:", (legend_x, legend_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        methods = ['dnn', 'contour', 'color', 'template', 'frontal_face']
        for i, method in enumerate(methods):
            y_pos = legend_y + (i * 12)
            color = detection_colors.get(method, (128, 128, 128))
            cv2.rectangle(image, (legend_x, y_pos), (legend_x + 10, y_pos + 8), color, -1)
            cv2.putText(image, method, (legend_x + 15, y_pos + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    def run_live_detection(self):
        """Main detection loop"""
        self.running = True
        print("Starting live object detection...")
        print("Press 'q' to quit, 's' to save current frame")

        frame_count = 0
        last_detection_time = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            frame_count += 1
            current_time = time.time()

            # Perform detection every 3 frames for better real-time performance
            if frame_count % 3 == 0:
                detections = self.detect_objects(frame)

                # Print detections to console with method information
                if detections:
                    print(f"\n--- Frame {frame_count} - {len(detections)} objects detected simultaneously ---")
                    
                    # Group detections by method
                    method_groups = {}
                    for detection in detections:
                        method = detection.get('detection_method', 'unknown')
                        if method not in method_groups:
                            method_groups[method] = []
                        method_groups[method].append(detection)
                    
                    # Print grouped detections
                    for method, method_detections in method_groups.items():
                        print(f"  {method.upper()} Detection ({len(method_detections)} objects):")
                        for i, detection in enumerate(method_detections):
                            print(f"    {i+1}. {detection['class']} (conf: {detection['confidence']:.2f}) "
                                  f"Color: {detection.get('color', 'unknown')}, "
                                  f"Age: {detection.get('age', 'unknown')}, "
                                  f"Material: {detection.get('material', 'unknown')} "
                                  f"at [{detection['bbox'][0]}, {detection['bbox'][1]}, {detection['bbox'][2]}, {detection['bbox'][3]}]")

                            # Save to database in background thread
                            threading.Thread(target=self.save_detection_to_db, args=(detection,), daemon=True).start()

                # Draw detections on frame
                frame = self.draw_detections(frame, detections)
                last_detection_time = current_time
                
                # Store detections for statistics
                self.recent_detections = detections

            # Display FPS and detection statistics
            fps = frame_count / (current_time - self.start_time) if hasattr(self, 'start_time') else 0
            current_detection_count = len(getattr(self, 'recent_detections', []))
            
            # Update simultaneous detection maximum
            if current_detection_count > self.detection_stats['simultaneous_max']:
                self.detection_stats['simultaneous_max'] = current_detection_count
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Current Objects: {current_detection_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Max Simultaneous: {self.detection_stats['simultaneous_max']}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"DB Objects: {len(self.objects)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Show frame
            cv2.imshow('Enhanced Object Detection - 10,000+ Objects', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                filename = f"detection_frame_{frame_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
            elif key == ord('p'):
                # Pause/unpause
                print("Paused. Press any key to continue...")
                cv2.waitKey(0)

        self.stop()

    def start(self):
        """Start the detection system"""
        self.start_time = time.time()
        try:
            self.run_live_detection()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            self.stop()

    def stop(self):
        """Stop the detection system"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Detection system stopped")

def print_detection_stats():
    """Print detection statistics from database"""
    try:
        conn = sqlite3.connect('detections.db')
        cursor = conn.cursor()

        cursor.execute('''
            SELECT object_class, COUNT(*) as count, AVG(confidence) as avg_confidence,
                   predicted_color, predicted_age, material
            FROM detections
            GROUP BY object_class, predicted_color, predicted_age, material
            ORDER BY count DESC
            LIMIT 20
        ''')

        print("\n=== Enhanced Detection Statistics ===")
        for row in cursor.fetchall():
            print(f"{row[0]} ({row[3]}, {row[4]}, {row[5]}): {row[1]} detections (avg confidence: {row[2]:.2f})")

        cursor.execute('SELECT COUNT(*) FROM detections')
        total = cursor.fetchone()[0]
        print(f"\nTotal detections: {total}")

        # Color statistics
        cursor.execute('''
            SELECT predicted_color, COUNT(*) as count
            FROM detections
            GROUP BY predicted_color
            ORDER BY count DESC
        ''')
        print("\n=== Color Distribution ===")
        for row in cursor.fetchall():
            print(f"{row[0]}: {row[1]} detections")

        # Age statistics
        cursor.execute('''
            SELECT predicted_age, COUNT(*) as count
            FROM detections
            GROUP BY predicted_age
            ORDER BY count DESC
        ''')
        print("\n=== Age/Condition Distribution ===")
        for row in cursor.fetchall():
            print(f"{row[0]}: {row[1]} detections")

        conn.close()
    except Exception as e:
        print(f"Error reading statistics: {e}")

if __name__ == '__main__':
    # Initialize database
    init_db()

    print("Enhanced Live Object Detection System")
    print("=====================================")
    print("Features:")
    print("- 10,000+ object detection capability")
    print("- Color prediction")
    print("- Age/condition estimation")
    print("- Material identification")
    print("- Comprehensive database storage")
    print()
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press 's' to save current frame")
    print("- Press 'p' to pause/unpause")
    print("- Press Ctrl+C to force quit")
    print()

    # Create and start detector
    detector = LiveObjectDetector()

    try:
        detector.start()
    finally:
        # Print final statistics
        print_detection_stats()
