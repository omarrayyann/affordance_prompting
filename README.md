# Affordance Prompting

**Examples:**

| Object  | Name | Result |
|:-------------:|:-----------:|:------:|
| Pot           | Pick        | ![pot](https://github.com/user-attachments/assets/47a2f807-16ee-4099-9630-35221eb79d47) |
| Door          | Open        | ![door](https://github.com/user-attachments/assets/dfc29bc8-ebab-41c1-8571-771219192d29) |
| Drawer          | Open        | ![drawer](https://github.com/user-attachments/assets/857453c6-8e85-456a-873c-5074a189f670) |



  
**How to use:**

1. Clone the repositry
```shell
git clone x
```

2. Install the required dependencies:
```python
pip install -r requirements.txt
```
3. Add your OpenAI API key to a file named `key.txt` in the main directory.

4. Run with a given image, object_name, action name
```python
python main.py --image <path_to_image> --object_name <object_name> --action <action>
```

5. Click on a chosen starting position in the image, then close the window.


