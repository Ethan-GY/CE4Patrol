class CE4Patrol:
    def __init__(self, vlm_model):
        self.vlm = vlm_model
        self.safety_rules = load_json("safety_rules.json")
        self.risk_level = load_json("risk_level.json")
    def inject_context(self, image, timestamp, gps, history):
        rule_prompt = f'according to {self.safety_rules},'
        spatial_prompt = f'current location is {gps}, time is {timestamp}, history is {history[-3:]}.'
        ref_image_prompt = f'normal image is {image}'
        full_prompt = f'[{rule_prompt}] + [{spatial_prompt}] + [{ref_image_prompt}], is there any anomaly in the picture?'
        return full_prompt
    def infer(self, image, metadata):
        prompt = self.inject_context(image, **metadata)
        response = self.vlm.generate(image,prompt)
        return parse_json_output(response)
