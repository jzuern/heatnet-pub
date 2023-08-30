import numpy as np
#import mapping

cpdef unsigned short[:,:] relabel_vistas_image(unsigned short[:,:] img, (int, int) relabel_dict):
    relabelled = np.zeros_like(img, dtype=np.uint16)

    cdef int x, y, h, w, current_p, current_class, current_id, id_counter
    id_counter = 0
    h = img.shape[0]
    w = img.shape[1]

    used_ids = {}

    for x in range(w):
        for y in range(h):
            current_p = int(img[y,x])
            current_class = int(current_p/256)

            if current_class in relabel_dict:
                current_id = int(current_p % 256)

                co = current_class*256 + current_id

                if co in used_ids.keys():
                    current_id = used_ids[co]
                else:
                    used_ids[co] = id_counter
                    current_id = id_counter
                    id_counter = id_counter + 1


                relabelled[y, x] = relabel_dict[current_class] * 256 + current_id
            else:
                relabelled[y, x] = 12 * 256

    return relabelled

cpdef unsigned char[:,:,:] relabel_image(unsigned char[:,:] seg,  dict relabel_dict):
    cdef int x, y, h, w, current_p, current_class, current_id, id_counter
    id_counter = 0
    h = seg.shape[0]
    w = seg.shape[1]
    relabelled = np.zeros((h, w, 3), dtype=np.uint8)

    used_ids = {}

    for x in range(w):
        for y in range(h):
            current_class = int(seg[y,x])

            if current_class in relabel_dict:
                color = relabel_dict[current_class]
                relabelled[y, x, 0] = color[0]
                relabelled[y, x, 1] = color[1]
                relabelled[y, x, 2] = color[2]


    return relabelled