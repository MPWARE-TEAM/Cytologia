def merge_adjacent_boxes(boxes):
    if not boxes:
        return []

    def overlap(box1, box2):
        return (
                box1[0] <= box2[1] and box2[0] <= box1[1] and
                box1[2] <= box2[3] and box2[2] <= box1[3]
        )

    def merge(box1, box2):
        return (
            min(box1[0], box2[0]),
            max(box1[1], box2[1]),
            min(box1[2], box2[2]),
            max(box1[3], box2[3]),
            0.5 * (box1[4] + box2[4])
        )

    merged = []
    while boxes:
        current = boxes.pop(0)
        merged.append(current)

        i = 0
        while i < len(boxes):
            if overlap(current, boxes[i]):
                current = merge(current, boxes.pop(i))
                merged[-1] = current
            else:
                i += 1

    return merged
