from collision_avoidance_space import collision_avoidance_space
from fake_bank_note_detection import fake_bank_note_detection


def run_collision_avoidance_space():
    path = input("Full path of dataset (leave empty for default path): ")
    print()
    if path == "":
        collision_avoidance_space()
    else:
        collision_avoidance_space(path=path)


def run_fake_bank_note_detection():
    train_path = input("Full path of train dataset (leave empty for default path): ")
    test_path = input("Full path of test dataset (leave empty for default path): ")
    print()
    if train_path != "" and test_path != "":
        fake_bank_note_detection(train_set_path=train_path, test_set_path=test_path)
    elif train_path != "" and test_path == "":
        fake_bank_note_detection(train_set_path=train_path)
    elif train_path == "" and test_path != "":
        fake_bank_note_detection(test_set_path=test_path)
    else:
        fake_bank_note_detection()


def main():
    print("Choose which exercise to run:")
    print("Collision avoidance in space - press 1")
    print("Detecting bank note fraud - press 2")
    answer = int(input("Choice: "))

    if answer == 1:
        run_collision_avoidance_space()
        if input("Run the other exercise? (y/n) ") == 'y':
            run_fake_bank_note_detection()
    else:
        run_fake_bank_note_detection()
        if input("Run the other exercise? (y/n) ") == 'y':
            run_collision_avoidance_space()
    print("END")


if __name__ == "__main__":
    main()
