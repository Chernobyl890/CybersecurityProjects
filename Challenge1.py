class LFSR: # representing the linear feedback shift register
    def __init__(self, feedback, init_value):
        self.feedback = feedback # feedback value
        self.state = init_value # initial value

    def step(self):  # step method updates LFSR's state according to the feedback
        feedback_bit = (self.state >> 31) & 1 # leftmost bit of the state
        new_bit = 0 # nitialize to store the new bit value
        # Iterate each bit of the feedback
        for i in range(32):  
            # check if i-th bit of the feedback is set
            if (self.feedback >> i) & 1:
                # XOR the i-th bit of the state with the new bit
                new_bit ^= (self.state >> i) & 1
                # shift the state one position left and set rightmost bit to the new bit
        self.state = ((self.state << 1) | new_bit) & 0xFFFFFFFF

    def generate_key(self): # generate_key method steps LFSR eight times and returns the lowest byte 
        for _ in range(8):
            self.step()
        return self.state & 0xFF


def encrypt(input_data, key_stream): # encrypt function takes the input data and a key stream generator as inputs
    encrypted_data = bytearray() # initialize empty byte array called encrypt_data
    for i in range(len(input_data)): # loop iterates over each index 'i' in the range of the input data
        encrypted_byte = input_data[i] ^ key_stream.generate_key() # XOR the i-th bit of the input data with the byte generated from the key stream
        encrypted_data.append(encrypted_byte)  # encrypted_byte obtained from the previous line is appended, building the encrypted data byte by byte 
    return encrypted_data # return the encrypted byte array


# Example usage
initial_value = 0xFFFFFFFF # set initial value to 0xFFFFFFFF (32 bit hex value)
feedback_value = 0x87654321 # set the feedback value to 0x87654321 (32 bit hex value) which is used in the step function to determine which bits are involved in the feedback op

input_data = bytearray(b"apple") # initializes the input data with a sequence of bytes (this will be encrypted using the LFSR-generated key stream)
lfsr = LFSR(feedback_value, initial_value) # create instance of the LFSR class using the feedback and initial values

encrypted_data = encrypt(input_data, lfsr) # call the encrypt function to encrypt the input data using the LFSR-generated key stream. Stored in the encrypted_data variable
print("Encrypted data:", encrypted_data) # print the encrypted data to the console. This will contain the encrypted bytes which are the result of applying the XOR operation between the original input data and the LFSR-generated key stream. 

# Second example test as seen in the dev challenge PDF reversing the encrypted bytes and returning 'apple' (for testing purposes)
input_data1 = bytearray(b"\x9e\x8f\x8f\x93\x9a")
lfsr1 = LFSR(feedback_value, initial_value)

encrypted_data1 = encrypt(input_data1, lfsr1)
print("Encrypted data:", encrypted_data1)
